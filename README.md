# Optiver-Realized-Volatility-Prediction
Github Repo for The Kaggle Competition

# Features To Add(from Trade Record)
- [ ] Price Resistance To No.Trades 价格对交易量的阻力
- [ ] Volume Of Trades From Time_ID 单个时段总交易量
- [ ] Price Level breakages (No. of) 整位价格突破（次数）
- [ ] Abnormal Price Surge/Plumb (No. of) 不正常价格下跌/上涨（次数）
- [ ] Total Large Orders (No. of) 总大宗交易数量
- [ ] Breaking Of Normal Trading Range (No. of) (Overall From A Single Stock) 正常交易范围突破次数


# 1. LSTM Volatility Over-value & Under-value Prediction
22/09/21
Because realized volatility(R.V.) exhibit **autocorrelation**, R.V. from the next 10-minute window can be viewed as the **R.V. from the current 10-minute window plus/minus a certain value.**

By plotting the relationship between **R.V.** and the difference between future **R.V.** and current **R.V.**(**R.V. Residual**), we get the following relationship.

<p align="center">
  <img src="https://github.com/ZhendongTian/Optiver-Realized-Volatility-Prediction/blob/main/gaussian.png" />
</p>
<p align="center">>
    <em>X-axis represents the value of R.V. and Y-axis represents R.V. Residual</em>
</p>

As can be seen from the relationship diagram, there exist a certain region where there is a slightly twisted **gaussian relationship**, we can therefore utilize that for correcting our **R.V.** **if we know if it's over-valued or under-valued** for values within that region.

## 1.1 Methodology

### 1.1.1 Dataset
The dataset we are using is the trade records for a 10-minute interval which contains **all the trades executed** during the 10-minute interval.
Every record shows the **time the trade** happens, the **price of the trade**, the **size** and **number of orders for the size**.

An slice of the trade record is shown in the table below.
|    time_id |   seconds_in_bucket |   price |   size |   order_count |   stock_id |
| ----------:|--------------------:|--------:|-------:|--------------:|-----------:|
|          5 |                  21 | 1.0023  |    326 |            12 |          0 |
|          5 |                  46 | 1.00278 |    128 |             4 |          0 |
|          5 |                  50 | 1.00282 |     55 |             1 |          0 |
|          5 |                  57 | 1.00316 |    121 |             5 |          0 |
|          5 |                  68 | 1.00365 |      4 |             1 |          0 |

**For the interests of using LSTM**, we are only using the time series of columns **price**, **size**, **order_count**. In this case, **second_in_bucket** becomes *trivial* because the concept of time will be incorported in the **LSTM** model.

### 1.1.2 Preprocessing
Not all 
