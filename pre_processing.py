import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import sort_edge_index
import os.path as osp
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from sklearn.preprocessing import StandardScaler


def _resolve_raw(raw_dir, candidates):
    """
    Return the first filename in ``candidates`` that exists in ``raw_dir``.
    Lets the loaders accept the Kaggle-native filenames (e.g.
    ``HI-Small_Trans.csv``) as well as the legacy ``*_raw.csv`` names without a
    manual rename. Falls back to the first (preferred) candidate so a missing
    file raises a clear error naming the expected file.
    """
    for name in candidates:
        if osp.exists(osp.join(raw_dir, name)):
            return name
    return candidates[0]


class EllipticDataset(InMemoryDataset):
    """
    Elliptic Bitcoin transaction graph. Nodes are transactions with 165
    anonymised features and a timestep; labels are licit (0) / illicit (1) /
    unknown (-1). The temporal split follows the standard Elliptic convention:
    timesteps 1-30 train, 31-40 validation, 41-49 test. Features are
    standardised with statistics fit on the training timesteps only, and the
    timestep itself is excluded from the feature matrix to avoid distribution
    shift between the disjoint train/val/test time ranges.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(EllipticDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            _resolve_raw(self.raw_dir, ['elliptic_txs_features.csv',
                                        'elliptic_txs_features_raw.csv']),
            _resolve_raw(self.raw_dir, ['elliptic_txs_classes.csv',
                                        'elliptic_txs_classes_raw.csv']),
            _resolve_raw(self.raw_dir, ['elliptic_txs_edgelist.csv',
                                        'elliptic_txs_edgelist_raw.csv']),
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        features_df = pd.read_csv(self.raw_paths[0], header=None)
        features_df.columns = ['txId'] + ['time_step'] + [f'V{i}' for i in range(1, 166)]
        classes_df = pd.read_csv(self.raw_paths[1])
        edgelist_df = pd.read_csv(self.raw_paths[2])

        # Remap string labels to integers (unknown -> -1).
        class_mapping = {'unknown': -1, '1': 1, '2': 0}
        classes_df['class'] = classes_df['class'].map(class_mapping)

        # Remap raw transaction ids to contiguous node indices 0..N-1.
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(features_df['txId'])}
        classes_df['txId'] = classes_df['txId'].map(node_mapping)
        edgelist_df['txId1'] = edgelist_df['txId1'].map(node_mapping)
        edgelist_df['txId2'] = edgelist_df['txId2'].map(node_mapping)
        edgelist_df = edgelist_df.dropna()  # drop edges referencing unmapped nodes
        classes_df = classes_df.sort_values('txId').set_index('txId')

        # Keep time_step out of the feature matrix: train/val/test have disjoint
        # timestep ranges, so using it as a feature causes distribution shift at test time.
        time_step_series = features_df['time_step'].values
        time_steps = torch.tensor(time_step_series, dtype=torch.long)

        # Build feature matrix (V1..V165) and standardise, fitting on training rows only.
        feature_array = features_df.drop(columns=['txId', 'time_step']).values.astype('float32')
        train_rows = time_step_series <= 30
        scaler = StandardScaler()
        scaler.fit(feature_array[train_rows])
        feature_array = scaler.transform(feature_array)

        features_tensor = torch.tensor(feature_array, dtype=torch.float)
        edge_index_directed = torch.tensor(edgelist_df.values.T, dtype=torch.long)
        edge_index_tensor = torch.cat(
            [edge_index_directed, edge_index_directed.flip(0)], dim=1
        )
        edge_index_tensor = sort_edge_index(
            edge_index_tensor, num_nodes=features_tensor.shape[0],
            sort_by_row=False,
        )
        y_tensor = torch.tensor(classes_df['class'].values, dtype=torch.long)

        data = Data(x=features_tensor, edge_index=edge_index_tensor, y=y_tensor)

        # Masks. The *_perf_eval_mask variants restrict to known-label nodes,
        # which is where performance is actually measured (the full masks are
        # used for message passing so unlabelled nodes still propagate features).
        known_nodes_mask = (data.y != -1)

        train_mask = (time_steps >= 1) & (time_steps <= 30)
        val_mask = (time_steps >= 31) & (time_steps <= 40)
        test_mask = (time_steps >= 41) & (time_steps <= 49)

        train_perf_eval_mask = train_mask & known_nodes_mask
        val_perf_eval_mask = val_mask & known_nodes_mask
        test_perf_eval_mask = test_mask & known_nodes_mask

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_perf_eval_mask = train_perf_eval_mask
        data.val_perf_eval_mask = val_perf_eval_mask
        data.test_perf_eval_mask = test_perf_eval_mask

        torch.save(self.collate([data]), self.processed_paths[0])


class IBMAMLDataset(InMemoryDataset):
    """
    Unified loader for the four IBM 'Transactions for AML' variants
    (HI/LI x Small/Medium). Each transaction is a node; a directed edge links
    transaction u -> v when the receiving account of u is the sending account
    of v and v occurs within a 4-hour window (``delta_minutes=240``) of u.
    Node features are the standardised transaction amounts, one-hot
    payment-format / currency fields, and day/hour/minute. The split is
    60/20/20 in chronological (timestamp-sorted) order.

    A single implementation replaces the previous four near-identical classes;
    only the raw filename and (for HI-Small) the ``read_csv`` dtype map differ,
    both captured in ``_VARIANTS``. The HI-Small dtype map is preserved exactly
    because reading the amount columns as float32 produces a marginally
    different StandardScaler fit than float64 — keeping it ensures the processed
    tensors are identical to those produced by the original per-variant classes.

    The thin ``IBMAMLDataset_*`` subclasses below preserve the original public
    API and per-variant processed-cache directories.
    """
    _VARIANTS = {
        "HiSmall":  {"prefix": "HI-Small",
                     "read_dtype": {"Amount Received": "float32",
                                    "Amount Paid": "float32",
                                    "class": "int8"}},
        "LiSmall":  {"prefix": "LI-Small",  "read_dtype": None},
        "HiMedium": {"prefix": "HI-Medium", "read_dtype": None},
        "LiMedium": {"prefix": "LI-Medium", "read_dtype": None},
    }

    def __init__(self, root, variant, transform=None, pre_transform=None, pre_filter=None):
        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown IBM-AML variant '{variant}'. "
                f"Choose from {list(self._VARIANTS)}."
            )
        self.variant = variant
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        prefix = self._VARIANTS[self.variant]["prefix"]
        return [_resolve_raw(self.raw_dir,
                             [f"{prefix}_Trans.csv", f"{prefix}_Trans_raw.csv"])]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _preprocess_ibm_edges(self, data_df, num_obs, delta_minutes=240):
        """
        Build transaction-to-transaction edges. Two transactions are linked when
        the receiving account of the earlier one is the sending account of the
        later one and they fall within ``delta_minutes`` of each other. The data
        is processed in 100 chronological pieces to bound memory.
        """
        print("Processing edges... This may take a while.")
        data_df_accounts = data_df[['txId', 'Account', 'Account.1', 'Timestamp']]

        source = []
        target = []

        pieces = 100
        for i in tqdm(range(pieces)):
            start = i * num_obs // pieces
            end = (i + 1) * num_obs // pieces
            data_df_right = data_df_accounts.iloc[start:end]

            if data_df_right.empty:
                continue

            min_timestamp = data_df_right['Timestamp'].min()
            max_timestamp = data_df_right['Timestamp'].max()

            delta = timedelta(minutes=delta_minutes)
            window_start = min_timestamp - delta

            data_df_left = data_df_accounts[
                (data_df_accounts['Timestamp'] >= window_start) &
                (data_df_accounts['Timestamp'] <= max_timestamp)
            ]

            # Link where recipient of a 'left' tx is the sender of a 'right' tx.
            data_df_join = data_df_left.merge(
                data_df_right,
                left_on='Account.1',
                right_on='Account',
                suffixes=('_1', '_2')
            )

            total_minutes = (
                data_df_join['Timestamp_2'] - data_df_join['Timestamp_1']
            ).dt.total_seconds() / 60.0
            mask = (total_minutes >= 0) & (total_minutes <= delta_minutes)
            source.extend(data_df_join.loc[mask, 'txId_1'].tolist())
            target.extend(data_df_join.loc[mask, 'txId_2'].tolist())

        df_edges = pd.DataFrame({'txId1': source, 'txId2': target})
        print(f"Edge processing complete. Found {len(df_edges)} edges.")
        return df_edges

    def process(self):
        read_dtype = self._VARIANTS[self.variant]["read_dtype"]

        print("Reading raw transaction data...")
        if read_dtype is not None:
            df_features = pd.read_csv(self.raw_paths[0], dtype=read_dtype)
        else:
            df_features = pd.read_csv(self.raw_paths[0])

        # 1. Parse timestamps, sort chronologically, drop self-transfers.
        date_format = '%Y/%m/%d %H:%M'
        df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format=date_format)
        df_features.sort_values('Timestamp', inplace=True)
        df_features = df_features[df_features['Account'] != df_features['Account.1']]

        # Use all transactions in the file (no subsetting).
        num_obs = len(df_features)

        # 2. Contiguous node ids 0..N-1.
        df_features.reset_index(drop=True, inplace=True)
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'txId'}, inplace=True)

        # 3. Name and select the columns we use.
        df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank',
                               'Account.1', 'Amount Received', 'Receiving Currency',
                               'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
        df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency',
                                   'Amount Paid', 'Payment Currency', 'Payment Format',
                                   'class', 'Account', 'Account.1']]

        # 4. Edges.
        df_edges = self._preprocess_ibm_edges(
            data_df=df_features[['txId', 'Account', 'Account.1', 'Timestamp']],
            num_obs=num_obs,
            delta_minutes=240
        )

        # 5. Feature engineering.
        print("Performing feature engineering...")
        timestamps = pd.DatetimeIndex(df_features['Timestamp'])
        df_features['Day'] = timestamps.day.astype('float32')
        df_features['Hour'] = timestamps.hour.astype('float32')
        df_features['Minute'] = timestamps.minute.astype('float32')

        df_features = df_features.drop(columns=['Timestamp', 'Account', 'Account.1'])

        df_features = pd.get_dummies(
            df_features,
            columns=['Receiving Currency', 'Payment Currency', 'Payment Format'],
            dtype='float32'
        )
        # Standardise the amount columns, fitting on the training rows only to
        # avoid leaking val/test statistics. df_features is timestamp-sorted and
        # the split is a chronological 60/20/20 (see the masks below), so the
        # training rows are the first 60% of the (sorted) dataframe.
        train_size = int(0.6 * num_obs)
        for amount_col in ['Amount Received', 'Amount Paid']:
            scaler = StandardScaler()
            scaler.fit(df_features[[amount_col]].iloc[:train_size])
            df_features[amount_col] = scaler.transform(df_features[[amount_col]])

        # 6. Tensors.
        y = torch.tensor(df_features['class'].values, dtype=torch.long)

        feature_cols = df_features.columns.drop(['txId', 'class'])
        x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)

        edge_index_directed = torch.tensor(
            df_edges[['txId1', 'txId2']].values, dtype=torch.long
        ).t().contiguous()
        edge_index = torch.cat(
            [edge_index_directed, edge_index_directed.flip(0)], dim=1
        )
        edge_index = sort_edge_index(edge_index, num_nodes=x.shape[0], sort_by_row=False)

        # 7. Masks (60/20/20, chronological).
        mask = torch.tensor([False] * num_obs)
        train_size = int(0.6 * num_obs)
        val_size = int(0.2 * num_obs)

        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True

        # 8. Save.
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")


class IBMAMLDataset_HiSmall(IBMAMLDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root, variant="HiSmall", **kwargs)


class IBMAMLDataset_LiSmall(IBMAMLDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root, variant="LiSmall", **kwargs)


class IBMAMLDataset_HiMedium(IBMAMLDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root, variant="HiMedium", **kwargs)


class IBMAMLDataset_LiMedium(IBMAMLDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root, variant="LiMedium", **kwargs)


class AMLSimDataset(InMemoryDataset):
    """
    AMLSim account graph. Nodes are accounts; each transaction is a directed
    edge sender -> receiver. An account is labelled illicit if it sent or
    received any fraudulent transaction. Features (initial balance and
    sent/received amount aggregates) are built point-in-time so that an
    account only "sees" transactions up to the end of its own split horizon,
    avoiding temporal leakage. The split is 60/20/20 by the time of each
    account's first transaction.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(AMLSimDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['accounts.csv', 'transactions.csv', 'alerts.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        import numpy as np

        accounts_df = pd.read_csv(self.raw_paths[0])
        transactions_df = pd.read_csv(self.raw_paths[1])
        alerts_df = pd.read_csv(self.raw_paths[2])

        # 1. Merge transactions with alerts to get fraud labels per transaction.
        txn = pd.merge(transactions_df, alerts_df, on='TX_ID', how='left')
        txn = txn.rename(columns={
            'SENDER_ACCOUNT_ID_x': 'SENDER_ACCOUNT',
            'RECEIVER_ACCOUNT_ID_x': 'RECEIVER_ACCOUNT',
            'TX_AMOUNT_x': 'TX_AMOUNT',
            'TIMESTAMP_x': 'TIMESTAMP',
            'IS_FRAUD_x': 'IS_FRAUD'
        })
        txn['IS_FRAUD'] = txn['IS_FRAUD'].fillna(0).astype(int)

        # Order transactions chronologically so the split and the feature
        # aggregation below are time-aware. TIMESTAMP may be an integer
        # simulation step or a date string; parse strings to datetime so they
        # sort chronologically (numeric stamps already sort correctly).
        if not pd.api.types.is_numeric_dtype(txn['TIMESTAMP']):
            txn['TIMESTAMP'] = pd.to_datetime(txn['TIMESTAMP'], errors='coerce')
        txn = txn.sort_values('TIMESTAMP', kind='stable').reset_index(drop=True)

        # 2. Account-to-index mapping (nodes = accounts).
        all_account_ids = sorted(accounts_df['ACCOUNT_ID'].unique())
        account_to_idx = {acc_id: idx for idx, acc_id in enumerate(all_account_ids)}
        num_accounts = len(all_account_ids)
        print(f"Number of account nodes: {num_accounts}")

        # 3. Account-level labels.
        fraud_senders = set(txn.loc[txn['IS_FRAUD'] == 1, 'SENDER_ACCOUNT'].unique())
        fraud_receivers = set(txn.loc[txn['IS_FRAUD'] == 1, 'RECEIVER_ACCOUNT'].unique())
        fraud_accounts = fraud_senders | fraud_receivers
        y = torch.zeros(num_accounts, dtype=torch.long)
        for acc_id in fraud_accounts:
            if acc_id in account_to_idx:
                y[account_to_idx[acc_id]] = 1
        print(f"Illicit accounts: {y.sum().item()} / {num_accounts} "
              f"({100 * y.sum().item() / num_accounts:.2f}%)")

        # 4. Edges: one directed edge per transaction.
        src = txn['SENDER_ACCOUNT'].map(account_to_idx).values
        dst = txn['RECEIVER_ACCOUNT'].map(account_to_idx).values
        edge_index_directed = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_index = torch.cat(
            [edge_index_directed, edge_index_directed.flip(0)], dim=1
        )
        edge_index = sort_edge_index(edge_index, num_nodes=num_accounts, sort_by_row=False)
        print(f"Number of edges (transactions): {edge_index.shape[1]}")

        # 5. Temporal 60/20/20 split on accounts, ordered by first-transaction time.
        sender_first = txn.groupby('SENDER_ACCOUNT')['TIMESTAMP'].min()
        receiver_first = txn.groupby('RECEIVER_ACCOUNT')['TIMESTAMP'].min()
        first_seen = pd.concat([sender_first, receiver_first]).groupby(level=0).min()
        first_seen = first_seen.reindex(all_account_ids)
        # Accounts that never transact have no timestamp; treat them as earliest
        # so they fall into the training split rather than the test horizon.
        first_seen = first_seen.fillna(txn['TIMESTAMP'].min())

        ordered_accounts = first_seen.sort_values(kind='stable').index.tolist()
        train_size = int(0.6 * num_accounts)
        val_size = int(0.2 * num_accounts)
        train_accounts = set(ordered_accounts[:train_size])
        val_accounts = set(ordered_accounts[train_size:train_size + val_size])
        test_accounts = set(ordered_accounts[train_size + val_size:])

        # Boundary timestamps between the splits (used to bound feature horizons).
        t_train = first_seen.loc[ordered_accounts[train_size]]
        t_val = first_seen.loc[ordered_accounts[train_size + val_size]]

        train_mask = torch.tensor([a in train_accounts for a in all_account_ids], dtype=torch.bool)
        val_mask = torch.tensor([a in val_accounts for a in all_account_ids], dtype=torch.bool)
        test_mask = torch.tensor([a in test_accounts for a in all_account_ids], dtype=torch.bool)
        for name, m in (('train', train_mask), ('val', val_mask), ('test', test_mask)):
            n = int(m.sum())
            pos = int(y[m].sum())
            print(f"  {name}: {n} accounts, {pos} illicit ({100 * pos / max(n, 1):.2f}%)")

        # 6. Point-in-time account features. Each split only aggregates
        #    transactions up to the end of its own horizon.
        accounts_sorted = accounts_df.set_index('ACCOUNT_ID').loc[all_account_ids]

        def _agg(sub_txn):
            sent = sub_txn.groupby('SENDER_ACCOUNT')['TX_AMOUNT'].agg(
                mean_amount_sent='mean', total_amount_sent='sum'
            )
            recv = sub_txn.groupby('RECEIVER_ACCOUNT')['TX_AMOUNT'].agg(
                mean_amount_received='mean', total_amount_received='sum'
            )
            return sent, recv

        train_sent, train_recv = _agg(txn[txn['TIMESTAMP'] < t_train])
        val_sent, val_recv = _agg(txn[txn['TIMESTAMP'] < t_val])
        test_sent, test_recv = _agg(txn)

        agg_cols = ['mean_amount_sent', 'total_amount_sent',
                    'mean_amount_received', 'total_amount_received']
        feat_df = pd.DataFrame({'ACCOUNT_ID': all_account_ids})
        feat_df = feat_df.merge(
            accounts_sorted[['INIT_BALANCE']].reset_index(),
            on='ACCOUNT_ID', how='left'
        )
        feat_df = feat_df.set_index('ACCOUNT_ID')
        for col in agg_cols:
            feat_df[col] = 0.0

        def _fill(accounts, sent, recv):
            acc_idx = feat_df.index.intersection(accounts)
            common_s = acc_idx.intersection(sent.index)
            feat_df.loc[common_s, ['mean_amount_sent', 'total_amount_sent']] = \
                sent.loc[common_s].values
            common_r = acc_idx.intersection(recv.index)
            feat_df.loc[common_r, ['mean_amount_received', 'total_amount_received']] = \
                recv.loc[common_r].values

        _fill(train_accounts, train_sent, train_recv)
        _fill(val_accounts, val_sent, val_recv)
        _fill(test_accounts, test_sent, test_recv)
        feat_df = feat_df.reset_index().fillna(0)

        # 7. Normalise numerical features (fit on train accounts only).
        num_cols = ['INIT_BALANCE'] + agg_cols
        train_rows = train_mask.numpy()
        scaler = StandardScaler()
        scaler.fit(feat_df.loc[train_rows, num_cols])
        feat_df[num_cols] = scaler.transform(feat_df[num_cols])

        feature_cols = [c for c in feat_df.columns if c != 'ACCOUNT_ID']
        x = torch.tensor(feat_df[feature_cols].values, dtype=torch.float)
        print(f"Feature matrix shape: {x.shape}  ({len(feature_cols)} features: {feature_cols})")

        # 8. Save.
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")
