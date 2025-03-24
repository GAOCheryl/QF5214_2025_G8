import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'


######
    def fit(self, dl_train, df_all, dl_valid=None):
        ######
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                ###### 6
            #    predictions, metrics = self.predict(dl_valid)
                predictions, metrics, real_returns, real_prices, market_cap = self.predict(dl_valid, df_all)
            
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
        
            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                break

    def predict(self, dl_test, df_all):
        if df_all is None or df_all.empty:
            raise ValueError("df_all is None or empty. Ensure data is loaded properly.")

        if isinstance(self.fitted, (int, float)) and self.fitted < 0:
            raise ValueError("Model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        preds, ic, ric, labels = [], [], [], []

        # Harmonize the index format of df_all.
        df_all["Date"] = pd.to_datetime(df_all["Date"])  # datetime type
        df_all = df_all.rename(columns={"Date": "datetime", "Ticker": "instrument"})
        
        # Re-indexing.
        original_prices = df_all.set_index(["datetime", "instrument"])["Close"] 
        original_market_caps = df_all.set_index(["datetime", "instrument"])["Market_Cap"]

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())
            labels.append(label.detach().cpu().numpy())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        real_returns = pd.Series(np.concatenate(labels), index=dl_test.get_index(), name="Actual_Return")

        # Checking for index matches.
        common_index = dl_test.get_index().intersection(original_prices.index)
        if common_index.empty:
            raise ValueError("No matching indices found between dl_test and df_all.")

        # Ensure use of raw data.
        real_prices = original_prices.loc[common_index].rename("Price")
        market_cap = original_market_caps.loc[common_index].rename("Market_Cap")

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic) / np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric) / np.std(ric)
        }

        return predictions, metrics, real_returns, real_prices, market_cap
