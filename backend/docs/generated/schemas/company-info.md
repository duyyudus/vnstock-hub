# company.info

- Class: `CompanyReference`
- Method: `info`
- Signature: `()`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, vci`
- Default route source: `vci`
- Default provider: `company.Company.overview`

Get company info/overview.

## Purpose

Get company info/overview.

## Parameters

_None._

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`

```text
symbol, business_model, history, charter_capital, num_employees, listing_date, founded_date, exchange, website, address, tax_id, phone, email
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `business_model` | `profile` |
| `history` | `history` |
| `charter_capital` | `charter_capital` |
| `num_employees` | `num_employees` |
| `listing_date` | `listing_date` |
| `founded_date` | `founded_date` |
| `exchange` | `exchange` |
| `website` | `website` |
| `address` | `address` |
| `tax_id` | `tax_id` |
| `phone` | `phone` |
| `email` | `email` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, short_name, exchange, sector, industry, profile, history, num_employees, founded_date, listing_date, charter_capital, issued_share, website, address, phone, email, tax_id
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:03.103508+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, sector, industry, profile, history, charter_capital, issued_share
```
- Dtypes: `{'symbol': 'str', 'sector': 'str', 'industry': 'str', 'profile': 'str', 'history': 'str', 'charter_capital': 'int64', 'issued_share': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "sector": "Ngân hàng",
    "industry": "Ngân hàng",
    "profile": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam (Vietcombank) chính thức đi vào hoạt động ngày 01/04/1963. Là ngân hàng thương mại nhà nước đầu tiên được Chính phủ lựa chọn thực hiện thí điểm cổ phần hoá, Ngân hàng Ngoại thương Việt Nam chính thức hoạt động với tư cách là một Ngân hàng Thương mại Cổ phần từ ngày 02/06/2008 sau khi thực hiện thành công kế hoạch cổ phần hóa thông qua việc phát hành cổ phiếu lần đầu ra công chúng. Năm 2024, so với cùng kỳ, biên lãi thuần (NIM) ở mức 2.86%, giảm 0.15%. Tỷ lệ nợ xấu ở mức 0.96%, giảm 0.02%. Tỷ lệ bao phủ nợ xấu ở mức 223.31%, giảm 6.99%. Lợi nhuận sau thuế công ty mẹ có giá trị bằng 33,8 nghìn tỷ đồng, tăng 2.42%. Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE) ở mức 18.74%, giảm 3.25%. VCB chính thức niêm yết và giao dịch trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh từ năm 2009.",
    "history": " - Ngày 30/10/1962: Ngân hàng Ngoại thương Việt Nam (Vietcombank) được thành lập có tiền thân là Cục Ngoại Hối trực thuộc Ngân Hàng Quốc Gia Việt Nam;  - Ngày 01/04/1963: Vietcombank chính thức đi vào hoạt động;  - Năm 1990: Vietcombank chuyển thành một NHTM nhà nước hoạt động đa năng;  - Năm 2007: Vietcombank và NHTMCP SeaBank ký kết Hợp đồng với đối tác Cardif thành lập Công ty TNHH Bảo hiểm Nhân thọ Vietcombank – Cardif (VCLI);  - Ngày 26/12/2007: Vietcombank phát hành đợt cổ phiếu đầu tiên ra công chúng.  - Năm 2008: Ngân hàng ngoại thương Việt Nam chính thức chuyển đổi thành Ngân Hàng TMCP Ngoại Thương Việt Nam;  - Năm 2009: Cổ phiếu của Vietcombank chính thức được niêm yết trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh (HOSE);  - Ngày 30/09/2011: Ngân hàng Mizuho (MHCB) đã chính thức trở thành nhà đầu tư chiến lược vào Vietcombank, nắm giữ 15% vốn điều lệ của Vietcombank;  - Ngày 16/01/2019: Tăng vốn điều lệ lên 37.088.774.480.000 đồng;  - Ngày 10/03/2022: Tăng vốn điều lệ lên 47.325.165.710.000 đồng;  - Ngày 05/10/2023: Tăng vốn điều lệ lên 55.890.912.620.000 đồng;  - Ngày 28/04/2025: Tăng vốn điều lệ lên 83.556.750.940.000 đồng do phát hành cổ phiếu trả cổ tức;",
    "charter_capital": 83556750940000,
    "issued_share": 8355675094
  }
]
```

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `overview`

```text
symbol, company_profile, history, charter_capital, icb_name3, icb_name2, issue_share, organ_name
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `company_profile` | `profile` |
| `history` | `history` |
| `charter_capital` | `charter_capital` |
| `icb_name3` | `industry` |
| `icb_name2` | `sector` |
| `issue_share` | `issued_share` |
| `organ_name` | `name` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, short_name, exchange, sector, industry, profile, history, num_employees, founded_date, listing_date, charter_capital, issued_share, website, address, phone, email, tax_id
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:01.722787+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, sector, industry, profile, history, charter_capital, issued_share
```
- Dtypes: `{'symbol': 'str', 'sector': 'str', 'industry': 'str', 'profile': 'str', 'history': 'str', 'charter_capital': 'int64', 'issued_share': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "sector": "Ngân hàng",
    "industry": "Ngân hàng",
    "profile": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam (Vietcombank) chính thức đi vào hoạt động ngày 01/04/1963. Là ngân hàng thương mại nhà nước đầu tiên được Chính phủ lựa chọn thực hiện thí điểm cổ phần hoá, Ngân hàng Ngoại thương Việt Nam chính thức hoạt động với tư cách là một Ngân hàng Thương mại Cổ phần từ ngày 02/06/2008 sau khi thực hiện thành công kế hoạch cổ phần hóa thông qua việc phát hành cổ phiếu lần đầu ra công chúng. Năm 2024, so với cùng kỳ, biên lãi thuần (NIM) ở mức 2.86%, giảm 0.15%. Tỷ lệ nợ xấu ở mức 0.96%, giảm 0.02%. Tỷ lệ bao phủ nợ xấu ở mức 223.31%, giảm 6.99%. Lợi nhuận sau thuế công ty mẹ có giá trị bằng 33,8 nghìn tỷ đồng, tăng 2.42%. Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE) ở mức 18.74%, giảm 3.25%. VCB chính thức niêm yết và giao dịch trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh từ năm 2009.",
    "history": " - Ngày 30/10/1962: Ngân hàng Ngoại thương Việt Nam (Vietcombank) được thành lập có tiền thân là Cục Ngoại Hối trực thuộc Ngân Hàng Quốc Gia Việt Nam;  - Ngày 01/04/1963: Vietcombank chính thức đi vào hoạt động;  - Năm 1990: Vietcombank chuyển thành một NHTM nhà nước hoạt động đa năng;  - Năm 2007: Vietcombank và NHTMCP SeaBank ký kết Hợp đồng với đối tác Cardif thành lập Công ty TNHH Bảo hiểm Nhân thọ Vietcombank – Cardif (VCLI);  - Ngày 26/12/2007: Vietcombank phát hành đợt cổ phiếu đầu tiên ra công chúng.  - Năm 2008: Ngân hàng ngoại thương Việt Nam chính thức chuyển đổi thành Ngân Hàng TMCP Ngoại Thương Việt Nam;  - Năm 2009: Cổ phiếu của Vietcombank chính thức được niêm yết trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh (HOSE);  - Ngày 30/09/2011: Ngân hàng Mizuho (MHCB) đã chính thức trở thành nhà đầu tư chiến lược vào Vietcombank, nắm giữ 15% vốn điều lệ của Vietcombank;  - Ngày 16/01/2019: Tăng vốn điều lệ lên 37.088.774.480.000 đồng;  - Ngày 10/03/2022: Tăng vốn điều lệ lên 47.325.165.710.000 đồng;  - Ngày 05/10/2023: Tăng vốn điều lệ lên 55.890.912.620.000 đồng;  - Ngày 28/04/2025: Tăng vốn điều lệ lên 83.556.750.940.000 đồng do phát hành cổ phiếu trả cổ tức;",
    "charter_capital": 83556750940000,
    "issued_share": 8355675094
  }
]
```
