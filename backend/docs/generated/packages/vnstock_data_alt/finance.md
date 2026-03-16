# Finance

- Qualified name: `app.lib.vnstock_data_alt.api.financial.Finance`
- Signature: `(source, symbol, period='quarter', get_all=True, show_log=False)`
- Supported sources: `kbs, mas, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### balance_sheet

- Kind: `method`
- Signature: `(period = None, limit = 12, include_metadata = False, display_mode = "<FieldDisplayMode.STD: 'std'>", show_log = False) -> DataFrame chứa bảng cân đối kế toán.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa bảng cân đối kế toán.`
- Purpose: Retrieve balance sheet data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `12` | `` |  | `5` |  | Số kỳ báo cáo tối đa cần lấy. Mặc định 4. |
| `include_metadata` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `omitted; default False` |  | Bao gồm thông tin audit và unit trong rows. Mặc định False. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.financial.Finance`
- Provider method: `balance_sheet`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:42.169658+00:00`
- Success: `True`
- Row count: `77`

```text
item, item_id, 2025, 2024, 2023, 2022, 2021
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64', '2021': 'float64'}`

```json
[
  {
    "item": "1. Các khoản phải thu",
    "item_id": "1_accounts_receivables",
    "2025": 18039522000.0,
    "2024": 14040294000.0,
    "2023": 11790173000.0,
    "2022": 24483406000.0,
    "2021": 15803945000.0
  },
  {
    "item": "1. Chứng khoán đầu tư sẵn sàng để bán",
    "item_id": "1_available_for_sales_securities",
    "2025": 143080817000.0,
    "2024": 86799901000.0,
    "2023": 67882480000.0,
    "2022": 100739670000.0,
    "2021": 71114698000.0
  },
  {
    "item": "1. Vốn của TCTD",
    "item_id": "1_capital",
    "2025": 89361977000.0,
    "2024": 61696139000.0,
    "2023": 61696139000.0,
    "2022": 53130392000.0,
    "2021": 42428821000.0
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.financial.Finance`
- Provider method: `balance_sheet`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:49.735148+00:00`
- Success: `True`
- Row count: `14`

```text
period, year_period, A. TÀI SẢN, I. Tiền mặt, vàng bạc, đá quý, II. Tiền gửi tại NHNN, III. Tiền, vàng gửi tại các TCTD khác và cho vay các TCTD khác, 1. Tiền, vàng gửi tại các TCTD khác, 2. Cho vay các TCTD khác, 3. Dự phòng rủi ro cho vay các TCTD khác, IV. Chứng khoán kinh doanh, 1. Chứng khoán kinh doanh, 2. Dự phòng giảm giá chứng khoán kinh doanh, V. Các công cụ tài chính phái sinh và các tài sản tài chính khác, VI. Cho vay khách hàng, 1. Cho vay và cho thuê tài chính khách hàng, 2. Dự phòng rủi ro cho vay và cho thuê tài chính khách hàng, VII. Hoạt động mua nợ, 1. Mua nợ, 2. Dự phòng rủi ro hoạt động mua nợ, VIII. Chứng khoán đầu tư, 1. Chứng khoán đầu tư sẵn sàng để bán, 2. Chứng khoán đầu tư giữ đến ngày đáo hạn, 3. Dự phòng giảm giá chứng khoán đầu tư, IX. Góp vốn, đầu tư dài hạn, 1. Đầu tư vào công ty con, 2. Đầu tư vào công ty liên doanh, liên kết, 3. Đầu tư dài hạn khác, 4. Dự phòng giảm giá đầu tư dài hạn, X. Tài sản cố định, 1. Tài sản cố định hữu hình, a. Nguyên giá TSCĐ, b. Hao mòn TSCĐ, 2. Tài sản cố định thuê tài chính, 3. Tài sản cố định vô hình, a. Nguyên giá BĐSĐT, b. Hao mòn BĐSĐT, XII. Tài sản "Có" khác, 1. Các khoản phải thu, 2. Các khoản lãi, phí phải thu, 3. Tài sản thuế TNDN hoãn lại, 4. Tài sản Có khác, - Trong đó: Lợi thế thương mại, 5. Các khoản dự phòng rủi ro cho các tài sản Có nội bảng khác, TỔNG CỘNG TÀI SẢN, B. NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU, I. Các khoản nợ Chính phủ và NHNN, II. Tiền gửi và vay các TCTD khác, 1. Tiền gửi của các TCTD khác, 2. Vay các TCTD khác, III. Tiền gửi của khách hàng, IV. Các công cụ tài chính phái sinh và các khoản nợ tài chính khác, V. Vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro, VI. Phát hành giấy tờ có giá, VII. Các khoản nợ khác, 1. Các khoản lãi, phí phải trả, 2. Thuế TNDN hoãn lại phải trả, 3. Các khoản phải trả và công nợ khác, 4. Dự phòng rủi ro khác (Dự phòng cho công nợ tiềm ẩn và cam kết ngoại bảng), TỔNG NỢ PHẢI TRẢ, VIII. Vốn và các quỹ, 1. Vốn của TCTD, a. Vốn điều lệ, b. Vốn đầu tư XDCB, c. Thặng dư vốn cổ phần, d. Cổ phiếu quỹ, e. Cổ phiếu ưu đãi, g. Vốn khác, 2. Quỹ của TCTD, 3. Chênh lệch tỷ giá hối đoái, 4. Chênh lệch đánh giá lại tài sản, 5. Lợi nhuận chưa phân phối/Lỗ lũy kế, 6. Lợi ích cổ đông không kiểm soát, IX. Lợi ích của cổ đông thiểu số, TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU, VII. Chứng khoán đầu tư, VIII. Góp vốn, đầu tư dài hạn, IX. Tài sản cố định, X. Bất động sản đầu tư, XI. Tài sản "Có" khác
```
- Dtypes: `{'period': 'int64', 'year_period': 'int64', 'A. TÀI SẢN': 'str', 'I. Tiền mặt, vàng bạc, đá quý': 'str', 'II. Tiền gửi tại NHNN': 'str', 'III. Tiền, vàng gửi tại các TCTD khác và cho vay các TCTD khác': 'str', '1. Tiền, vàng gửi tại các TCTD khác': 'str', '2. Cho vay các TCTD khác': 'str', '3. Dự phòng rủi ro cho vay các TCTD khác': 'str', 'IV. Chứng khoán kinh doanh': 'str', '1. Chứng khoán kinh doanh': 'str', '2. Dự phòng giảm giá chứng khoán kinh doanh': 'str', 'V. Các công cụ tài chính phái sinh và các tài sản tài chính khác': 'str', 'VI. Cho vay khách hàng': 'str', '1. Cho vay và cho thuê tài chính khách hàng': 'str', '2. Dự phòng rủi ro cho vay và cho thuê tài chính khách hàng': 'str', 'VII. Hoạt động mua nợ': 'str', '1. Mua nợ': 'str', '2. Dự phòng rủi ro hoạt động mua nợ': 'str', 'VIII. Chứng khoán đầu tư': 'str', '1. Chứng khoán đầu tư sẵn sàng để bán': 'str', '2. Chứng khoán đầu tư giữ đến ngày đáo hạn': 'str', '3. Dự phòng giảm giá chứng khoán đầu tư': 'str', 'IX. Góp vốn, đầu tư dài hạn': 'str', '1. Đầu tư vào công ty con': 'str', '2. Đầu tư vào công ty liên doanh, liên kết': 'str', '3. Đầu tư dài hạn khác': 'str', '4. Dự phòng giảm giá đầu tư dài hạn': 'str', 'X. Tài sản cố định': 'str', '1. Tài sản cố định hữu hình': 'str', 'a. Nguyên giá TSCĐ': 'str', 'b. Hao mòn TSCĐ': 'str', '2. Tài sản cố định thuê tài chính': 'str', '3. Tài sản cố định vô hình': 'str', 'a. Nguyên giá BĐSĐT': 'str', 'b. Hao mòn BĐSĐT': 'str', 'XII. Tài sản "Có" khác': 'str', '1. Các khoản phải thu': 'str', '2. Các khoản lãi, phí phải thu': 'str', '3. Tài sản thuế TNDN hoãn lại': 'str', '4. Tài sản Có khác': 'str', '- Trong đó: Lợi thế thương mại': 'str', '5. Các khoản dự phòng rủi ro cho các tài sản Có nội bảng khác': 'str', 'TỔNG CỘNG TÀI SẢN': 'str', 'B. NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU': 'str', 'I. Các khoản nợ Chính phủ và NHNN': 'str', 'II. Tiền gửi và vay các TCTD khác': 'str', '1. Tiền gửi của các TCTD khác': 'str', '2. Vay các TCTD khác': 'str', 'III. Tiền gửi của khách hàng': 'str', 'IV. Các công cụ tài chính phái sinh và các khoản nợ tài chính khác': 'str', 'V. Vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro': 'str', 'VI. Phát hành giấy tờ có giá': 'str', 'VII. Các khoản nợ khác': 'str', '1. Các khoản lãi, phí phải trả': 'str', '2. Thuế TNDN hoãn lại phải trả': 'str', '3. Các khoản phải trả và công nợ khác': 'str', '4. Dự phòng rủi ro khác (Dự phòng cho công nợ tiềm ẩn và cam kết ngoại bảng)': 'str', 'TỔNG NỢ PHẢI TRẢ': 'str', 'VIII. Vốn và các quỹ': 'str', '1. Vốn của TCTD': 'str', 'a. Vốn điều lệ': 'str', 'b. Vốn đầu tư XDCB': 'str', 'c. Thặng dư vốn cổ phần': 'str', 'd. Cổ phiếu quỹ': 'str', 'e. Cổ phiếu ưu đãi': 'str', 'g. Vốn khác': 'str', '2. Quỹ của TCTD': 'str', '3. Chênh lệch tỷ giá hối đoái': 'str', '4. Chênh lệch đánh giá lại tài sản': 'str', '5. Lợi nhuận chưa phân phối/Lỗ lũy kế': 'str', '6. Lợi ích cổ đông không kiểm soát': 'str', 'IX. Lợi ích của cổ đông thiểu số': 'str', 'TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU': 'str', 'VII. Chứng khoán đầu tư': 'str', 'VIII. Góp vốn, đầu tư dài hạn': 'str', 'IX. Tài sản cố định': 'str', 'X. Bất động sản đầu tư': 'str', 'XI. Tài sản "Có" khác': 'str'}`

```json
[
  {
    "period": 2025,
    "year_period": 2025,
    "A. TÀI SẢN": NaN,
    "I. Tiền mặt, vàng bạc, đá quý": "15542768000000",
    "II. Tiền gửi tại NHNN": "37445504000000",
    "III. Tiền, vàng gửi tại các TCTD khác và cho vay các TCTD khác": "521938509000000",
    "1. Tiền, vàng gửi tại các TCTD khác": "515052787000000",
    "2. Cho vay các TCTD khác": "6885722000000",
    "3. Dự phòng rủi ro cho vay các TCTD khác": NaN,
    "IV. Chứng khoán kinh doanh": "11479097000000",
    "1. Chứng khoán kinh doanh": "11546520000000",
    "2. Dự phòng giảm giá chứng khoán kinh doanh": "-67423000000",
    "V. Các công cụ tài chính phái sinh và các tài sản tài chính khác": "374918000000",
    "VI. Cho vay khách hàng": "1648557141000000",
    "1. Cho vay và cho thuê tài chính khách hàng": "1673525675000000",
    "2. Dự phòng rủi ro cho vay và cho thuê tài chính khách hàng": "-24968534000000",
    "VII. Hoạt động mua nợ": NaN,
    "1. Mua nợ": NaN,
    "2. Dự phòng rủi ro hoạt động mua nợ": NaN,
    "VIII. Chứng khoán đầu tư": "162104164000000",
    "1. Chứng khoán đầu tư sẵn sàng để bán": "143080817000000",
    "2. Chứng khoán đầu tư giữ đến ngày đáo hạn": "22384962000000",
    "3. Dự phòng giảm giá chứng khoán đầu tư": "-3361615000000",
    "IX. Góp vốn, đầu tư dài hạn": "2260728000000",
    "1. Đầu tư vào công ty con": NaN,
    "2. Đầu tư vào công ty liên doanh, liên kết": "746639000000",
    "3. Đầu tư dài hạn khác": "1589089000000",
    "4. Dự phòng giảm giá đầu tư dài hạn": "-75000000000",
    "X. Tài sản cố định": "8232904000000",
    "1. Tài sản cố định hữu hình": "5618792000000",
    "a. Nguyên giá TSCĐ": "5284812000000",
    "b. Hao mòn TSCĐ": "-2670700000000",
    "2. Tài sản cố định thuê tài chính": NaN,
    "3. Tài sản cố định vô hình": "2614112000000",
    "a. Nguyên giá BĐSĐT": NaN,
    "b. Hao mòn BĐSĐT": NaN,
    "XII. Tài sản \"Có\" khác": "33993212000000",
    "1. Các khoản phải thu": "18039522000000",
    "2. Các khoản lãi, phí phải thu": "10007220000000",
    "3. Tài sản thuế TNDN hoãn lại": "13072000000",
    "4. Tài sản Có khác": "5951104000000",
    "- Trong đó: Lợi thế thương mại": NaN,
    "5. Các khoản dự phòng rủi ro cho các tài sản Có nội bảng khác": "-17706000000",
    "TỔNG CỘNG TÀI SẢN": "2441928945000000",
    "B. NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": NaN,
    "I. Các khoản nợ Chính phủ và NHNN": "160128325000000",
    "II. Tiền gửi và vay các TCTD khác": "321158844000000",
    "1. Tiền gửi của các TCTD khác": "305903589000000",
    "2. Vay các TCTD khác": "15255255000000",
    "III. Tiền gửi của khách hàng": "1672534103000000",
    "IV. Các công cụ tài chính phái sinh và các khoản nợ tài chính khác": NaN,
    "V. Vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": NaN,
    "VI. Phát hành giấy tờ có giá": "27101221000000",
    "VII. Các khoản nợ khác": "33470576000000",
    "1. Các khoản lãi, phí phải trả": "15457973000000",
    "2. Thuế TNDN hoãn lại phải trả": NaN,
    "3. Các khoản phải trả và công nợ khác": "18012603000000",
    "4. Dự phòng rủi ro khác (Dự phòng cho công nợ tiềm ẩn và cam kết ngoại bảng)": NaN,
    "TỔNG NỢ PHẢI TRẢ": "2214393069000000",
    "VIII. Vốn và các quỹ": "227535876000000",
    "1. Vốn của TCTD": "89361977000000",
    "a. Vốn điều lệ": "83556751000000",
    "b. Vốn đầu tư XDCB": NaN,
    "c. Thặng dư vốn cổ phần": "4995389000000",
    "d. Cổ phiếu quỹ": NaN,
    "e. Cổ phiếu ưu đãi": NaN,
    "g. Vốn khác": "809837000000",
    "2. Quỹ của TCTD": "36993479000000",
    "3. Chênh lệch tỷ giá hối đoái": "-918673000000",
    "4. Chênh lệch đánh giá lại tài sản": NaN,
    "5. Lợi nhuận chưa phân phối/Lỗ lũy kế": "102027572000000",
    "6. Lợi ích cổ đông không kiểm soát": "71521000000",
    "IX. Lợi ích của cổ đông thiểu số": NaN,
    "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": "2441928945000000",
    "VII. Chứng khoán đầu tư": NaN,
    "VIII. Góp vốn, đầu tư dài hạn": NaN,
    "IX. Tài sản cố định": NaN,
    "X. Bất động sản đầu tư": NaN,
    "XI. Tài sản \"Có\" khác": NaN
  },
  {
    "period": 2024,
    "year_period": 2024,
    "A. TÀI SẢN": NaN,
    "I. Tiền mặt, vàng bạc, đá quý": "14268064000000",
    "II. Tiền gửi tại NHNN": "49340493000000",
    "III. Tiền, vàng gửi tại các TCTD khác và cho vay các TCTD khác": "389951898000000",
    "1. Tiền, vàng gửi tại các TCTD khác": "384031890000000",
    "2. Cho vay các TCTD khác": "6920008000000",
    "3. Dự phòng rủi ro cho vay các TCTD khác": "-1000000000000",
    "IV. Chứng khoán kinh doanh": "4876237000000",
    "1. Chứng khoán kinh doanh": "4908527000000",
    "2. Dự phòng giảm giá chứng khoán kinh doanh": "-32290000000",
    "V. Các công cụ tài chính phái sinh và các tài sản tài chính khác": "1314434000000",
    "VI. Cho vay khách hàng": "1418015724000000",
    "1. Cho vay và cho thuê tài chính khách hàng": "1449198899000000",
    "2. Dự phòng rủi ro cho vay và cho thuê tài chính khách hàng": "-31183175000000",
    "VII. Hoạt động mua nợ": NaN,
    "1. Mua nợ": NaN,
    "2. Dự phòng rủi ro hoạt động mua nợ": NaN,
    "VIII. Chứng khoán đầu tư": "167383349000000",
    "1. Chứng khoán đầu tư sẵn sàng để bán": "86799901000000",
    "2. Chứng khoán đầu tư giữ đến ngày đáo hạn": "80829540000000",
    "3. Dự phòng giảm giá chứng khoán đầu tư": "-246092000000",
    "IX. Góp vốn, đầu tư dài hạn": "2228098000000",
    "1. Đầu tư vào công ty con": NaN,
    "2. Đầu tư vào công ty liên doanh, liên kết": "774176000000",
    "3. Đầu tư dài hạn khác": "1528922000000",
    "4. Dự phòng giảm giá đầu tư dài hạn": "-75000000000",
    "X. Tài sản cố định": "8092877000000",
    "1. Tài sản cố định hữu hình": "5530579000000",
    "a. Nguyên giá TSCĐ": "5072735000000",
    "b. Hao mòn TSCĐ": "-2510437000000",
    "2. Tài sản cố định thuê tài chính": NaN,
    "3. Tài sản cố định vô hình": "2562298000000",
    "a. Nguyên giá BĐSĐT": NaN,
    "b. Hao mòn BĐSĐT": NaN,
    "XII. Tài sản \"Có\" khác": "30402348000000",
    "1. Các khoản phải thu": "14040294000000",
    "2. Các khoản lãi, phí phải thu": "8868303000000",
    "3. Tài sản thuế TNDN hoãn lại": "991748000000",
    "4. Tài sản Có khác": "6516040000000",
    "- Trong đó: Lợi thế thương mại": NaN,
    "5. Các khoản dự phòng rủi ro cho các tài sản Có nội bảng khác": "-14037000000",
    "TỔNG CỘNG TÀI SẢN": "2085873522000000",
    "B. NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": NaN,
    "I. Các khoản nợ Chính phủ và NHNN": "78237337000000",
    "II. Tiền gửi và vay các TCTD khác": "234533958000000",
    "1. Tiền gửi của các TCTD khác": "223171381000000",
    "2. Vay các TCTD khác": "11362577000000",
    "III. Tiền gửi của khách hàng": "1514664850000000",
    "IV. Các công cụ tài chính phái sinh và các khoản nợ tài chính khác": NaN,
    "V. Vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": "529000000",
    "VI. Phát hành giấy tờ có giá": "24125059000000",
    "VII. Các khoản nợ khác": "38102621000000",
    "1. Các khoản lãi, phí phải trả": "13990276000000",
    "2. Thuế TNDN hoãn lại phải trả": NaN,
    "3. Các khoản phải trả và công nợ khác": "24112345000000",
    "4. Dự phòng rủi ro khác (Dự phòng cho công nợ tiềm ẩn và cam kết ngoại bảng)": NaN,
    "TỔNG NỢ PHẢI TRẢ": "1889664354000000",
    "VIII. Vốn và các quỹ": "196209168000000",
    "1. Vốn của TCTD": "61696139000000",
    "a. Vốn điều lệ": "55890913000000",
    "b. Vốn đầu tư XDCB": NaN,
    "c. Thặng dư vốn cổ phần": "4995389000000",
    "d. Cổ phiếu quỹ": NaN,
    "e. Cổ phiếu ưu đãi": NaN,
    "g. Vốn khác": "809837000000",
    "2. Quỹ của TCTD": "37052974000000",
    "3. Chênh lệch tỷ giá hối đoái": "-968292000000",
    "4. Chênh lệch đánh giá lại tài sản": NaN,
    "5. Lợi nhuận chưa phân phối/Lỗ lũy kế": "98332086000000",
    "6. Lợi ích cổ đông không kiểm soát": "96261000000",
    "IX. Lợi ích của cổ đông thiểu số": NaN,
    "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": "2085873522000000",
    "VII. Chứng khoán đầu tư": NaN,
    "VIII. Góp vốn, đầu tư dài hạn": NaN,
    "IX. Tài sản cố định": NaN,
    "X. Bất động sản đầu tư": NaN,
    "XI. Tài sản \"Có\" khác": NaN
  },
  {
    "period": 2023,
    "year_period": 2023,
    "A. TÀI SẢN": NaN,
    "I. Tiền mặt, vàng bạc, đá quý": "14504849000000",
    "II. Tiền gửi tại NHNN": "58104503000000",
    "III. Tiền, vàng gửi tại các TCTD khác và cho vay các TCTD khác": "336468607000000",
    "1. Tiền, vàng gửi tại các TCTD khác": "312001875000000",
    "2. Cho vay các TCTD khác": "30175707000000",
    "3. Dự phòng rủi ro cho vay các TCTD khác": "-5708975000000",
    "IV. Chứng khoán kinh doanh": "2495408000000",
    "1. Chứng khoán kinh doanh": "2511395000000",
    "2. Dự phòng giảm giá chứng khoán kinh doanh": "-15987000000",
    "V. Các công cụ tài chính phái sinh và các tài sản tài chính khác": NaN,
    "VI. Cho vay khách hàng": "1241675333000000",
    "1. Cho vay và cho thuê tài chính khách hàng": "1270359018000000",
    "2. Dự phòng rủi ro cho vay và cho thuê tài chính khách hàng": "-28683685000000",
    "VII. Hoạt động mua nợ": NaN,
    "1. Mua nợ": NaN,
    "2. Dự phòng rủi ro hoạt động mua nợ": NaN,
    "VIII. Chứng khoán đầu tư": NaN,
    "1. Chứng khoán đầu tư sẵn sàng để bán": "67882480000000",
    "2. Chứng khoán đầu tư giữ đến ngày đáo hạn": "78009747000000",
    "3. Dự phòng giảm giá chứng khoán đầu tư": "-112160000000",
    "IX. Góp vốn, đầu tư dài hạn": NaN,
    "1. Đầu tư vào công ty con": NaN,
    "2. Đầu tư vào công ty liên doanh, liên kết": "838225000000",
    "3. Đầu tư dài hạn khác": "1529145000000",
    "4. Dự phòng giảm giá đầu tư dài hạn": "-142425000000",
    "X. Tài sản cố định": NaN,
    "1. Tài sản cố định hữu hình": "5115612000000",
    "a. Nguyên giá TSCĐ": "4906881000000",
    "b. Hao mòn TSCĐ": "-2314312000000",
    "2. Tài sản cố định thuê tài chính": NaN,
    "3. Tài sản cố định vô hình": "2592569000000",
    "a. Nguyên giá BĐSĐT": NaN,
    "b. Hao mòn BĐSĐT": NaN,
    "XII. Tài sản \"Có\" khác": NaN,
    "1. Các khoản phải thu": "11790173000000",
    "2. Các khoản lãi, phí phải thu": "9200022000000",
    "3. Tài sản thuế TNDN hoãn lại": "848268000000",
    "4. Tài sản Có khác": "8828305000000",
    "- Trong đó: Lợi thế thương mại": NaN,
    "5. Các khoản dự phòng rủi ro cho các tài sản Có nội bảng khác": "-15463000000",
    "TỔNG CỘNG TÀI SẢN": "1839613198000000",
    "B. NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": NaN,
    "I. Các khoản nợ Chính phủ và NHNN": "1670837000000",
    "II. Tiền gửi và vay các TCTD khác": "213838980000000",
    "1. Tiền gửi của các TCTD khác": "193963218000000",
    "2. Vay các TCTD khác": "19875762000000",
    "III. Tiền gửi của khách hàng": "1395697611000000",
    "IV. Các công cụ tài chính phái sinh và các khoản nợ tài chính khác": "117752000000",
    "V. Vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": "365000000",
    "VI. Phát hành giấy tờ có giá": "19912623000000",
    "VII. Các khoản nợ khác": "43362364000000",
    "1. Các khoản lãi, phí phải trả": "19527028000000",
    "2. Thuế TNDN hoãn lại phải trả": NaN,
    "3. Các khoản phải trả và công nợ khác": "23835336000000",
    "4. Dự phòng rủi ro khác (Dự phòng cho công nợ tiềm ẩn và cam kết ngoại bảng)": NaN,
    "TỔNG NỢ PHẢI TRẢ": "1674600532000000",
    "VIII. Vốn và các quỹ": "165012666000000",
    "1. Vốn của TCTD": "61696139000000",
    "a. Vốn điều lệ": "55890913000000",
    "b. Vốn đầu tư XDCB": NaN,
    "c. Thặng dư vốn cổ phần": "4995389000000",
    "d. Cổ phiếu quỹ": NaN,
    "e. Cổ phiếu ưu đãi": NaN,
    "g. Vốn khác": "809837000000",
    "2. Quỹ của TCTD": "27447116000000",
    "3. Chênh lệch tỷ giá hối đoái": "-983237000000",
    "4. Chênh lệch đánh giá lại tài sản": NaN,
    "5. Lợi nhuận chưa phân phối/Lỗ lũy kế": "76758658000000",
    "6. Lợi ích cổ đông không kiểm soát": "93990000000",
    "IX. Lợi ích của cổ đông thiểu số": NaN,
    "TỔNG NỢ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU": "1839613198000000",
    "VII. Chứng khoán đầu tư": "145780067000000",
    "VIII. Góp vốn, đầu tư dài hạn": "2224945000000",
    "IX. Tài sản cố định": "7708181000000",
    "X. Bất động sản đầu tư": NaN,
    "XI. Tài sản \"Có\" khác": "30651305000000"
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.financial.Finance`
- Provider method: `balance_sheet`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:51.610109+00:00`
- Success: `True`
- Row count: `8`

```text
report_period, ticker, Cash and precious metals, Fixed assets, Tangible fixed assets, Cost, _Cost, __Cost, Accumulated depreciation, _Accumulated depreciation, __Accumulated depreciation, Finance leased assets, ___Cost, ____Cost, _____Cost, ___Accumulated depreciation, ____Accumulated depreciation, _____Accumulated depreciation, Intangible fixed assets, ______Cost, _______Cost, ________Cost, Accumulated amortization, Investment properties, Historical cost, ______Accumulated depreciation, _______Accumulated depreciation, ________Accumulated depreciation, Investment in other entities and long term investments, Investment in associate companies, Other Long-term investments, TOTAL ASSETS, TOTAL LIABILITIES, OWNER'S EQUITY, Charter capital, Capital surplus, Owner's other capital, Treasury shares, Difference upon assets revaluation, Foreign currency difference reserve, Retained Earnings, Minority interest (Before 2015), LIABILITIES AND SHAREHOLDER'S EQUITY, Balances with the SBV, Placements with and loans to other credit institutions, Trading securities, net, Trading securities, Derivatives and other financial assets, Loans and advances to customers, net, Loans and advances to customers, Investment securities, Available-for-sales securities, Held-to-maturity securities, Other assets, _Other assets, Due to Gov and Loans from SBV, Deposits and Loans from other credit institutions, _Deposits and Loans from other credit institutions, Deposits from customers, Derivatives and other financial liabilities, Funds received from Gov, international and other institutions, Other liabilities, _Other liabilities, Capital, Fund for basic construction, Preferred shares, Reserves, Other guarantee, Foreign exchange transactions commitments, Foreign exchange buying commitments, Foreign exchange selling commitments, Swap commitments, Future commitments, Irrevocable loan commitments, Letters of credit commitments, Other commitments, Minority interest, Balances with other credit institutions, Loans to other credit institutions, Allowance for balances with and loans to other credit institutions, Debt Buying, _Debt Buying, __Debt Buying, ___Debt Buying, Other receivables, Accrued interest and fee receivables, Deferred tax assets, __Other assets, ___Other assets, In which: Goodwill, __Deposits and Loans from other credit institutions, ___Deposits and Loans from other credit institutions, Loans from other credit institutions, Accrued interest and fee payables, Deferred tax liabilities, __Other liabilities, ___Other liabilities, Allowance for other liabilities, Investments in joint-venture, Investments in associates
```
- Dtypes: `{'report_period': 'str', 'ticker': 'str', 'Cash and precious metals': 'float64', 'Fixed assets': 'float64', 'Tangible fixed assets': 'float64', 'Cost': 'float64', '_Cost': 'float64', '__Cost': 'float64', 'Accumulated depreciation': 'float64', '_Accumulated depreciation': 'float64', '__Accumulated depreciation': 'float64', 'Finance leased assets': 'float64', '___Cost': 'float64', '____Cost': 'float64', '_____Cost': 'float64', '___Accumulated depreciation': 'float64', '____Accumulated depreciation': 'float64', '_____Accumulated depreciation': 'float64', 'Intangible fixed assets': 'float64', '______Cost': 'float64', '_______Cost': 'float64', '________Cost': 'float64', 'Accumulated amortization': 'float64', 'Investment properties': 'float64', 'Historical cost': 'float64', '______Accumulated depreciation': 'float64', '_______Accumulated depreciation': 'float64', '________Accumulated depreciation': 'float64', 'Investment in other entities and long term investments': 'float64', 'Investment in associate companies': 'float64', 'Other Long-term investments': 'float64', 'TOTAL ASSETS': 'float64', 'TOTAL LIABILITIES': 'float64', "OWNER'S EQUITY": 'float64', 'Charter capital': 'float64', 'Capital surplus': 'float64', "Owner's other capital": 'float64', 'Treasury shares': 'float64', 'Difference upon assets revaluation': 'float64', 'Foreign currency difference reserve': 'float64', 'Retained Earnings': 'float64', 'Minority interest (Before 2015)': 'float64', "LIABILITIES AND SHAREHOLDER'S EQUITY": 'float64', 'Balances with the SBV': 'float64', 'Placements with and loans to other credit institutions': 'float64', 'Trading securities, net': 'float64', 'Trading securities': 'float64', 'Derivatives and other financial assets': 'float64', 'Loans and advances to customers, net': 'float64', 'Loans and advances to customers': 'float64', 'Investment securities': 'float64', 'Available-for-sales securities': 'float64', 'Held-to-maturity securities': 'float64', 'Other assets': 'float64', '_Other assets': 'float64', 'Due to Gov and Loans from SBV': 'float64', 'Deposits and Loans from other credit institutions': 'float64', '_Deposits and Loans from other credit institutions': 'float64', 'Deposits from customers': 'float64', 'Derivatives and other financial liabilities': 'float64', 'Funds received from Gov, international and other institutions': 'float64', 'Other liabilities': 'float64', '_Other liabilities': 'float64', 'Capital': 'float64', 'Fund for basic construction': 'float64', 'Preferred shares': 'float64', 'Reserves': 'float64', 'Other guarantee': 'float64', 'Foreign exchange transactions commitments': 'float64', 'Foreign exchange buying commitments': 'float64', 'Foreign exchange selling commitments': 'float64', 'Swap commitments': 'float64', 'Future commitments': 'float64', 'Irrevocable loan commitments': 'float64', 'Letters of credit commitments': 'float64', 'Other commitments': 'float64', 'Minority interest': 'float64', 'Balances with other credit institutions': 'float64', 'Loans to other credit institutions': 'float64', 'Allowance for balances with and loans to other credit institutions': 'float64', 'Debt Buying': 'float64', '_Debt Buying': 'float64', '__Debt Buying': 'float64', '___Debt Buying': 'float64', 'Other receivables': 'float64', 'Accrued interest and fee receivables': 'float64', 'Deferred tax assets': 'float64', '__Other assets': 'float64', '___Other assets': 'float64', 'In which: Goodwill': 'float64', '__Deposits and Loans from other credit institutions': 'float64', '___Deposits and Loans from other credit institutions': 'float64', 'Loans from other credit institutions': 'float64', 'Accrued interest and fee payables': 'float64', 'Deferred tax liabilities': 'float64', '__Other liabilities': 'float64', '___Other liabilities': 'float64', 'Allowance for other liabilities': 'float64', 'Investments in joint-venture': 'float64', 'Investments in associates': 'float64'}`

```json
[
  {
    "report_period": "year",
    "ticker": "VCB",
    "Cash and precious metals": 12792045000000.0,
    "Fixed assets": 6527466000000.0,
    "Tangible fixed assets": 4459292000000.0,
    "Cost": 10534068000000.0,
    "_Cost": 0.0,
    "__Cost": 2772517000000.0,
    "Accumulated depreciation": -6074776000000.0,
    "_Accumulated depreciation": 0.0,
    "__Accumulated depreciation": 0.0,
    "Finance leased assets": 0.0,
    "___Cost": 10534068000000.0,
    "____Cost": 0.0,
    "_____Cost": 2772517000000.0,
    "___Accumulated depreciation": -6074776000000.0,
    "____Accumulated depreciation": 0.0,
    "_____Accumulated depreciation": 0.0,
    "Intangible fixed assets": 2068174000000.0,
    "______Cost": 10534068000000.0,
    "_______Cost": 0.0,
    "________Cost": 2772517000000.0,
    "Accumulated amortization": -704343000000.0,
    "Investment properties": 0.0,
    "Historical cost": 0.0,
    "______Accumulated depreciation": -6074776000000.0,
    "_______Accumulated depreciation": 0.0,
    "________Accumulated depreciation": 0.0,
    "Investment in other entities and long term investments": 2476067000000.0,
    "Investment in associate companies": 907647000000.0,
    "Other Long-term investments": 1635418000000.0,
    "TOTAL ASSETS": 1074026560000000.0,
    "TOTAL LIABILITIES": 1011847181000000.0,
    "OWNER'S EQUITY": 62179379000000.0,
    "Charter capital": 35977686000000.0,
    "Capital surplus": 0.0,
    "Owner's other capital": 344657000000.0,
    "Treasury shares": 0.0,
    "Difference upon assets revaluation": 119178000000.0,
    "Foreign currency difference reserve": 84450000000.0,
    "Retained Earnings": 16138687000000.0,
    "Minority interest (Before 2015)": 0.0,
    "LIABILITIES AND SHAREHOLDER'S EQUITY": 1074026560000000.0,
    "Balances with the SBV": 10845701000000.0,
    "Placements with and loans to other credit institutions": 250228037000000.0,
    "Trading securities, net": 2654806000000.0,
    "Trading securities": 2725051000000.0,
    "Derivatives and other financial assets": 275983000000.0,
    "Loans and advances to customers, net": 621573249000000.0,
    "Loans and advances to customers": 631866758000000.0,
    "Investment securities": 149296430000000.0,
    "Available-for-sales securities": 35321259000000.0,
    "Held-to-maturity securities": 114251030000000.0,
    "Other assets": 17356776000000.0,
    "_Other assets": 5879141000000.0,
    "Due to Gov and Loans from SBV": 90685315000000.0,
    "Deposits and Loans from other credit institutions": 76524079000000.0,
    "_Deposits and Loans from other credit institutions": 75245679000000.0,
    "Deposits from customers": 801929115000000.0,
    "Derivatives and other financial liabilities": 0.0,
    "Funds received from Gov, international and other institutions": 25803000000.0,
    "Other liabilities": 21221737000000.0,
    "_Other liabilities": 12484902000000.0,
    "Capital": 36322343000000.0,
    "Fund for basic construction": 0.0,
    "Preferred shares": 0.0,
    "Reserves": 9445732000000.0,
    "Other guarantee": 54250031000000.0,
    "Foreign exchange transactions commitments": 276512000000.0,
    "Foreign exchange buying commitments": 12471111000000.0,
    "Foreign exchange selling commitments": 49360171000000.0,
    "Swap commitments": 0.0,
    "Future commitments": 0.0,
    "Irrevocable loan commitments": 0.0,
    "Letters of credit commitments": 57703713000000.0,
    "Other commitments": 295856000000.0,
    "Minority interest": 68989000000.0,
    "Balances with other credit institutions": 187352500000000.0,
    "Loans to other credit institutions": 63875537000000.0,
    "Allowance for balances with and loans to other credit institutions": -1000000000000.0,
    "Debt Buying": 0.0,
    "_Debt Buying": 0.0,
    "__Debt Buying": 0.0,
    "___Debt Buying": 0.0,
    "Other receivables": 4065268000000.0,
    "Accrued interest and fee receivables": 7409149000000.0,
    "Deferred tax assets": 6740000000.0,
    "__Other assets": 17356776000000.0,
    "___Other assets": 5879141000000.0,
    "In which: Goodwill": 0.0,
    "__Deposits and Loans from other credit institutions": 76524079000000.0,
    "___Deposits and Loans from other credit institutions": 75245679000000.0,
    "Loans from other credit institutions": 1278400000000.0,
    "Accrued interest and fee payables": 8717540000000.0,
    "Deferred tax liabilities": 19295000000.0,
    "__Other liabilities": 21221737000000.0,
    "___Other liabilities": 12484902000000.0,
    "Allowance for other liabilities": 0.0,
    "Investments in joint-venture": 897308000000.0,
    "Investments in associates": 10339000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Cash and precious metals": 13778358000000.0,
    "Fixed assets": 6706503000000.0,
    "Tangible fixed assets": 4445709000000.0,
    "Cost": 11162170000000.0,
    "_Cost": 0.0,
    "__Cost": 3050669000000.0,
    "Accumulated depreciation": -6716461000000.0,
    "_Accumulated depreciation": 0.0,
    "__Accumulated depreciation": 0.0,
    "Finance leased assets": 0.0,
    "___Cost": 11162170000000.0,
    "____Cost": 0.0,
    "_____Cost": 3050669000000.0,
    "___Accumulated depreciation": -6716461000000.0,
    "____Accumulated depreciation": 0.0,
    "_____Accumulated depreciation": 0.0,
    "Intangible fixed assets": 2260794000000.0,
    "______Cost": 11162170000000.0,
    "_______Cost": 0.0,
    "________Cost": 3050669000000.0,
    "Accumulated amortization": -789875000000.0,
    "Investment properties": 0.0,
    "Historical cost": 0.0,
    "______Accumulated depreciation": -6716461000000.0,
    "_______Accumulated depreciation": 0.0,
    "________Accumulated depreciation": 0.0,
    "Investment in other entities and long term investments": 2464493000000.0,
    "Investment in associate companies": 951670000000.0,
    "Other Long-term investments": 1587823000000.0,
    "TOTAL ASSETS": 1222813692000000.0,
    "TOTAL LIABILITIES": 1141859355000000.0,
    "OWNER'S EQUITY": 80954337000000.0,
    "Charter capital": 37088774000000.0,
    "Capital surplus": 4995389000000.0,
    "Owner's other capital": 344658000000.0,
    "Treasury shares": 0.0,
    "Difference upon assets revaluation": 113011000000.0,
    "Foreign currency difference reserve": 16361000000.0,
    "Retained Earnings": 26126544000000.0,
    "Minority interest (Before 2015)": 0.0,
    "LIABILITIES AND SHAREHOLDER'S EQUITY": 1222813692000000.0,
    "Balances with the SBV": 34684091000000.0,
    "Placements with and loans to other credit institutions": 249470372000000.0,
    "Trading securities, net": 1801126000000.0,
    "Trading securities": 1889628000000.0,
    "Derivatives and other financial assets": 98312000000.0,
    "Loans and advances to customers, net": 724473254000000.0,
    "Loans and advances to customers": 734706891000000.0,
    "Investment securities": 167529689000000.0,
    "Available-for-sales securities": 35699090000000.0,
    "Held-to-maturity securities": 132271302000000.0,
    "Other assets": 21807494000000.0,
    "_Other assets": 4510592000000.0,
    "Due to Gov and Loans from SBV": 92365806000000.0,
    "Deposits and Loans from other credit institutions": 73617085000000.0,
    "_Deposits and Loans from other credit institutions": 71046512000000.0,
    "Deposits from customers": 928450869000000.0,
    "Derivatives and other financial liabilities": 0.0,
    "Funds received from Gov, international and other institutions": 20431000000.0,
    "Other liabilities": 26021232000000.0,
    "_Other liabilities": 15635924000000.0,
    "Capital": 42428821000000.0,
    "Fund for basic construction": 0.0,
    "Preferred shares": 0.0,
    "Reserves": 12186141000000.0,
    "Other guarantee": 53115849000000.0,
    "Foreign exchange transactions commitments": 292563000000.0,
    "Foreign exchange buying commitments": 25775812000000.0,
    "Foreign exchange selling commitments": 65818817000000.0,
    "Swap commitments": 0.0,
    "Future commitments": 0.0,
    "Irrevocable loan commitments": 0.0,
    "Letters of credit commitments": 57345298000000.0,
    "Other commitments": 349311000000.0,
    "Minority interest": 83459000000.0,
    "Balances with other credit institutions": 190100329000000.0,
    "Loans to other credit institutions": 62370043000000.0,
    "Allowance for balances with and loans to other credit institutions": -3000000000000.0,
    "Debt Buying": 0.0,
    "_Debt Buying": 0.0,
    "__Debt Buying": 0.0,
    "___Debt Buying": 0.0,
    "Other receivables": 8830390000000.0,
    "Accrued interest and fee receivables": 8064808000000.0,
    "Deferred tax assets": 405543000000.0,
    "__Other assets": 21807494000000.0,
    "___Other assets": 4510592000000.0,
    "In which: Goodwill": 0.0,
    "__Deposits and Loans from other credit institutions": 73617085000000.0,
    "___Deposits and Loans from other credit institutions": 71046512000000.0,
    "Loans from other credit institutions": 2570573000000.0,
    "Accrued interest and fee payables": 10363285000000.0,
    "Deferred tax liabilities": 22023000000.0,
    "__Other liabilities": 26021232000000.0,
    "___Other liabilities": 15635924000000.0,
    "Allowance for other liabilities": 0.0,
    "Investments in joint-venture": 940807000000.0,
    "Investments in associates": 10863000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Cash and precious metals": 15095394000000.0,
    "Fixed assets": 8539362000000.0,
    "Tangible fixed assets": 5411139000000.0,
    "Cost": 12866189000000.0,
    "_Cost": 0.0,
    "__Cost": 4211880000000.0,
    "Accumulated depreciation": -7455050000000.0,
    "_Accumulated depreciation": 0.0,
    "__Accumulated depreciation": 0.0,
    "Finance leased assets": 0.0,
    "___Cost": 12866189000000.0,
    "____Cost": 0.0,
    "_____Cost": 4211880000000.0,
    "___Accumulated depreciation": -7455050000000.0,
    "____Accumulated depreciation": 0.0,
    "_____Accumulated depreciation": 0.0,
    "Intangible fixed assets": 3128223000000.0,
    "______Cost": 12866189000000.0,
    "_______Cost": 0.0,
    "________Cost": 4211880000000.0,
    "Accumulated amortization": -1083657000000.0,
    "Investment properties": 0.0,
    "Historical cost": 0.0,
    "______Accumulated depreciation": -7455050000000.0,
    "_______Accumulated depreciation": 0.0,
    "________Accumulated depreciation": 0.0,
    "Investment in other entities and long term investments": 2239006000000.0,
    "Investment in associate companies": 726183000000.0,
    "Other Long-term investments": 1587823000000.0,
    "TOTAL ASSETS": 1326230092000000.0,
    "TOTAL LIABILITIES": 1232135113000000.0,
    "OWNER'S EQUITY": 94094979000000.0,
    "Charter capital": 37088774000000.0,
    "Capital surplus": 4995389000000.0,
    "Owner's other capital": 344658000000.0,
    "Treasury shares": 0.0,
    "Difference upon assets revaluation": 0.0,
    "Foreign currency difference reserve": 5103000000.0,
    "Retained Earnings": 36650228000000.0,
    "Minority interest (Before 2015)": 0.0,
    "LIABILITIES AND SHAREHOLDER'S EQUITY": 1326230092000000.0,
    "Balances with the SBV": 33139373000000.0,
    "Placements with and loans to other credit institutions": 267969645000000.0,
    "Trading securities, net": 1954061000000.0,
    "Trading securities": 1991861000000.0,
    "Derivatives and other financial assets": 0.0,
    "Loans and advances to customers, net": 820545467000000.0,
    "Loans and advances to customers": 839788261000000.0,
    "Investment securities": 156931097000000.0,
    "Available-for-sales securities": 42148831000000.0,
    "Held-to-maturity securities": 115382544000000.0,
    "Other assets": 19816687000000.0,
    "_Other assets": 5036638000000.0,
    "Due to Gov and Loans from SBV": 41176995000000.0,
    "Deposits and Loans from other credit institutions": 103583833000000.0,
    "_Deposits and Loans from other credit institutions": 100916433000000.0,
    "Deposits from customers": 1032113567000000.0,
    "Derivatives and other financial liabilities": 52031000000.0,
    "Funds received from Gov, international and other institutions": 14679000000.0,
    "Other liabilities": 33953811000000.0,
    "_Other liabilities": 24155977000000.0,
    "Capital": 42428821000000.0,
    "Fund for basic construction": 0.0,
    "Preferred shares": 0.0,
    "Reserves": 14925803000000.0,
    "Other guarantee": 50892327000000.0,
    "Foreign exchange transactions commitments": 654296000000.0,
    "Foreign exchange buying commitments": 53984032000000.0,
    "Foreign exchange selling commitments": 62672880000000.0,
    "Swap commitments": 0.0,
    "Future commitments": 0.0,
    "Irrevocable loan commitments": 0.0,
    "Letters of credit commitments": 45980494000000.0,
    "Other commitments": 680372000000.0,
    "Minority interest": 85024000000.0,
    "Balances with other credit institutions": 204713783000000.0,
    "Loans to other credit institutions": 64255862000000.0,
    "Allowance for balances with and loans to other credit institutions": -1000000000000.0,
    "Debt Buying": 0.0,
    "_Debt Buying": 0.0,
    "__Debt Buying": 0.0,
    "___Debt Buying": 0.0,
    "Other receivables": 6668595000000.0,
    "Accrued interest and fee receivables": 7206125000000.0,
    "Deferred tax assets": 909263000000.0,
    "__Other assets": 19816687000000.0,
    "___Other assets": 5036638000000.0,
    "In which: Goodwill": 0.0,
    "__Deposits and Loans from other credit institutions": 103583833000000.0,
    "___Deposits and Loans from other credit institutions": 100916433000000.0,
    "Loans from other credit institutions": 2667400000000.0,
    "Accrued interest and fee payables": 9797834000000.0,
    "Deferred tax liabilities": 0.0,
    "__Other liabilities": 33953811000000.0,
    "___Other liabilities": 24155977000000.0,
    "Allowance for other liabilities": 0.0,
    "Investments in joint-venture": 714935000000.0,
    "Investments in associates": 11248000000.0
  }
]
```

#### Notes / caveats

Retrieve balance sheet data.
Forwards supported kwargs (e.g., period, lang, dropna, show_log).

### cash_flow

- Kind: `method`
- Signature: `(period = None, limit = 12, include_metadata = False, display_mode = "<FieldDisplayMode.STD: 'std'>", show_log = False) -> DataFrame chứa báo cáo lưu chuyển tiền tệ.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa báo cáo lưu chuyển tiền tệ.`
- Purpose: Retrieve cash flow statement data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `12` | `` |  | `5` |  | Số kỳ báo cáo tối đa cần lấy. Mặc định 4. |
| `include_metadata` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `omitted; default False` |  | Bao gồm thông tin audit và unit trong rows. Mặc định False. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.financial.Finance`
- Provider method: `cash_flow`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:53.167470+00:00`
- Success: `True`
- Row count: `50`

```text
item, item_id, 2025, 2024, 2023, 2022, 2020
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64', '2020': 'float64'}`

```json
[
  {
    "item": "10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán",
    "item_id": "10_increase_decrease_in_trading_securities_and_investment_securities",
    "2025": -4474331000.0,
    "2024": -24134346000.0,
    "2023": 19792999000.0,
    "2022": 5431755000.0,
    "2020": 10336784000.0
  },
  {
    "item": "11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác",
    "item_id": "11_increase_decrease_in_derivatives_and_other_financial_assets",
    "2025": 939516000.0,
    "2024": -1314434000.0,
    "2023": 156515000.0,
    "2022": 146687000.0,
    "2020": 98312000.0
  },
  {
    "item": "12. (Tăng)/Giảm các khoản cho vay khách hàng",
    "item_id": "12_increase_decrease_in_loans_and_advances_to_customers",
    "2025": -224326776000.0,
    "2024": -178839881000.0,
    "2023": -125292768000.0,
    "2022": -184316295000.0,
    "2020": -105081370000.0
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.financial.Finance`
- Provider method: `cash_flow`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:56.358065+00:00`
- Success: `True`
- Row count: `14`

```text
period, year_period, Lưu chuyển tiền từ hoạt động kinh doanh, 1. Thu nhập lãi và các khoản thu nhập tương tự nhận được, 2. Chi phí lãi và các chi phí tương tự đã trả, 3. Thu nhập từ hoạt động dịch vụ nhận được, 4- Chênh lệch số tiền thực thu/thực chi từ hoạt động kinh doanh (ngoại tệ, vàng bạc, chứng khoán), 5. Thu nhập khác, 6. Tiền thu các khoản nợ đã được xử lý xóa, bù đắp bằng nguồn rủi ro, 7. Tiền chi trả cho nhân viên và hoạt động quản lý, công vụ, 8. Tiền thuế thu nhập thực nộp trong kỳ, Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động, Những thay đổi về tài sản hoạt động, 9. (Tăng)/Giảm các khoản tiền, vàng gửi và cho vay các TCTD khác, 10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán, 11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác, 12. (Tăng)/Giảm các khoản cho vay khách hàng, 13. Giảm nguồn dự phòng để bù đắp tổn thất các khoản, 14. (Tăng)/Giảm khác về tài sản hoạt động, Những thay đổi về công nợ hoạt động, 15. Tăng/(Giảm) các khoản nợ chính phủ và NHNN, 16. Tăng/(Giảm) các khoản tiền gửi, tiền vay các tổ chức tín dụng, 17. Tăng/(Giảm) tiền gửi của khách hàng (bao gồm cả Kho bạc Nhà nước), 18. Tăng/(Giảm) phát hành giấy tờ có giá (ngoại trừ giấy tờ có giá phát hành được tình vào hoạt động tài chính), 19. Tăng/(Giảm) vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro, 20. Tăng/(Giảm) các công cụ tài chính phái sinh và các khoản nợ tài chính khác, 21. Tăng/(Giảm) khác về công nợ hoạt động, 22. Chi từ các quỹ của TCTD, I - Lưu chuyển tiền thuần từ hoạt động kinh doanh, Lưu chuyển tiền từ hoạt động đầu tư, 1. Mua sắm tài sản cố định, 2. Tiền thu từ thanh lý, nhượng bán TSCĐ, 3. Tiền chi từ thanh lý, nhượng bán TSCĐ, 4. Mua sắm bất động sản đầu tư, 5. Tiền thu từ bán, thanh lý bất động sản đầu tư, 6. Tiền chi ra do bán, thanh lý bất động sản đầu tư, 7. Tiền chi đầu tư, góp vốn vào các đơn vị khác (mua công ty con, góp vốn liên doanh, liên kết, đầu tư dài hạn khác), 8. Tiền thu đầu tư, góp vốn vào các đơn vị khác (bán, thanh lý Công ty con, góp vốn liên doanh,  liên kết, đầu tư dài hạn khác), 9. Tiền thu cổ tức và lợi nhuận được chia từ các khoản đầu tư, góp vốn dài hạn, II- Lưu chuyển tiền thuần từ hoạt động đầu tư, Lưu chuyển tiền từ hoạt động tài chính, 2. Tiền thu từ phát hành giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác, 3. Tiền chi thanh toán giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác, 4. Cổ tức trả cho cổ đông, lợi nhuận đã chia, 5. Tiền chi ra mua cổ phiếu quỹ, 6. Tiền thu được do bán cổ phiếu quỹ, III- Lưu chuyển tiền thuần từ hoạt động tài chính, IV- Lưu chuyển tiền thuần trong kỳ, V- Tiền và các khoản tương đương tiền tại thời điểm đầu kỳ, VI- Điều chỉnh ảnh hưởng của thay đổi tỷ giá, VII. Tiền và các khoản tương đương tiền tại thời điểm cuối kỳ, 1. Tăng v ốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu
```
- Dtypes: `{'period': 'int64', 'year_period': 'int64', 'Lưu chuyển tiền từ hoạt động kinh doanh': 'str', '1. Thu nhập lãi và các khoản thu nhập tương tự nhận được': 'str', '2. Chi phí lãi và các chi phí tương tự đã trả': 'str', '3. Thu nhập từ hoạt động dịch vụ nhận được': 'str', '4- Chênh lệch số tiền thực thu/thực chi từ hoạt động kinh doanh (ngoại tệ, vàng bạc, chứng khoán)': 'str', '5. Thu nhập khác': 'str', '6. Tiền thu các khoản nợ đã được xử lý xóa, bù đắp bằng nguồn rủi ro': 'str', '7. Tiền chi trả cho nhân viên và hoạt động quản lý, công vụ': 'str', '8. Tiền thuế thu nhập thực nộp trong kỳ': 'str', 'Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động': 'str', 'Những thay đổi về tài sản hoạt động': 'str', '9. (Tăng)/Giảm các khoản tiền, vàng gửi và cho vay các TCTD khác': 'str', '10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán': 'str', '11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác': 'str', '12. (Tăng)/Giảm các khoản cho vay khách hàng': 'str', '13. Giảm nguồn dự phòng để bù đắp tổn thất các khoản': 'str', '14. (Tăng)/Giảm khác về tài sản hoạt động': 'str', 'Những thay đổi về công nợ hoạt động': 'str', '15. Tăng/(Giảm) các khoản nợ chính phủ và NHNN': 'str', '16. Tăng/(Giảm) các khoản tiền gửi, tiền vay các tổ chức tín dụng': 'str', '17. Tăng/(Giảm) tiền gửi của khách hàng (bao gồm cả Kho bạc Nhà nước)': 'str', '18. Tăng/(Giảm) phát hành giấy tờ có giá (ngoại trừ giấy tờ có giá phát hành được tình vào hoạt động tài chính)': 'str', '19. Tăng/(Giảm) vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro': 'str', '20. Tăng/(Giảm) các công cụ tài chính phái sinh và các khoản nợ tài chính khác': 'str', '21. Tăng/(Giảm) khác về công nợ hoạt động': 'str', '22. Chi từ các quỹ của TCTD': 'str', 'I - Lưu chuyển tiền thuần từ hoạt động kinh doanh': 'str', 'Lưu chuyển tiền từ hoạt động đầu tư': 'str', '1. Mua sắm tài sản cố định': 'str', '2. Tiền thu từ thanh lý, nhượng bán TSCĐ': 'str', '3. Tiền chi từ thanh lý, nhượng bán TSCĐ': 'str', '4. Mua sắm bất động sản đầu tư': 'str', '5. Tiền thu từ bán, thanh lý bất động sản đầu tư': 'str', '6. Tiền chi ra do bán, thanh lý bất động sản đầu tư': 'str', '7. Tiền chi đầu tư, góp vốn vào các đơn vị khác (mua công ty con, góp vốn liên doanh, liên kết, đầu tư dài hạn khác)': 'str', '8. Tiền thu đầu tư, góp vốn vào các đơn vị khác (bán, thanh lý Công ty con, góp vốn liên doanh,  liên kết, đầu tư dài hạn khác)': 'str', '9. Tiền thu cổ tức và lợi nhuận được chia từ các khoản đầu tư, góp vốn dài hạn': 'str', 'II- Lưu chuyển tiền thuần từ hoạt động đầu tư': 'str', 'Lưu chuyển tiền từ hoạt động tài chính': 'str', '2. Tiền thu từ phát hành giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác': 'str', '3. Tiền chi thanh toán giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác': 'str', '4. Cổ tức trả cho cổ đông, lợi nhuận đã chia': 'str', '5. Tiền chi ra mua cổ phiếu quỹ': 'str', '6. Tiền thu được do bán cổ phiếu quỹ': 'str', 'III- Lưu chuyển tiền thuần từ hoạt động tài chính': 'str', 'IV- Lưu chuyển tiền thuần trong kỳ': 'str', 'V- Tiền và các khoản tương đương tiền tại thời điểm đầu kỳ': 'str', 'VI- Điều chỉnh ảnh hưởng của thay đổi tỷ giá': 'str', 'VII. Tiền và các khoản tương đương tiền tại thời điểm cuối kỳ': 'str', '1. Tăng v ốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu': 'str'}`

```json
[
  {
    "period": 2025,
    "year_period": 2025,
    "Lưu chuyển tiền từ hoạt động kinh doanh": NaN,
    "1. Thu nhập lãi và các khoản thu nhập tương tự nhận được": "104279621000000",
    "2. Chi phí lãi và các chi phí tương tự đã trả": "-44982266000000",
    "3. Thu nhập từ hoạt động dịch vụ nhận được": "3413845000000",
    "4- Chênh lệch số tiền thực thu/thực chi từ hoạt động kinh doanh (ngoại tệ, vàng bạc, chứng khoán)": "5074170000000",
    "5. Thu nhập khác": "-340406000000",
    "6. Tiền thu các khoản nợ đã được xử lý xóa, bù đắp bằng nguồn rủi ro": "3916056000000",
    "7. Tiền chi trả cho nhân viên và hoạt động quản lý, công vụ": "-23796526000000",
    "8. Tiền thuế thu nhập thực nộp trong kỳ": "-9241967000000",
    "Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động": "38322527000000",
    "Những thay đổi về tài sản hoạt động": NaN,
    "9. (Tăng)/Giảm các khoản tiền, vàng gửi và cho vay các TCTD khác": "-10181043000000",
    "10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán": "-4474331000000",
    "11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác": "939516000000",
    "12. (Tăng)/Giảm các khoản cho vay khách hàng": "-224326776000000",
    "13. Giảm nguồn dự phòng để bù đắp tổn thất các khoản": "-7287782000000",
    "14. (Tăng)/Giảm khác về tài sản hoạt động": "-3414751000000",
    "Những thay đổi về công nợ hoạt động": NaN,
    "15. Tăng/(Giảm) các khoản nợ chính phủ và NHNN": "81890988000000",
    "16. Tăng/(Giảm) các khoản tiền gửi, tiền vay các tổ chức tín dụng": "86624886000000",
    "17. Tăng/(Giảm) tiền gửi của khách hàng (bao gồm cả Kho bạc Nhà nước)": "157869253000000",
    "18. Tăng/(Giảm) phát hành giấy tờ có giá (ngoại trừ giấy tờ có giá phát hành được tình vào hoạt động tài chính)": "2976037000000",
    "19. Tăng/(Giảm) vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": "-529000000",
    "20. Tăng/(Giảm) các công cụ tài chính phái sinh và các khoản nợ tài chính khác": NaN,
    "21. Tăng/(Giảm) khác về công nợ hoạt động": "-971727000000",
    "22. Chi từ các quỹ của TCTD": "-2625051000000",
    "I - Lưu chuyển tiền thuần từ hoạt động kinh doanh": "115341217000000",
    "Lưu chuyển tiền từ hoạt động đầu tư": NaN,
    "1. Mua sắm tài sản cố định": "-1453488000000",
    "2. Tiền thu từ thanh lý, nhượng bán TSCĐ": "17231000000",
    "3. Tiền chi từ thanh lý, nhượng bán TSCĐ": "-1288000000",
    "4. Mua sắm bất động sản đầu tư": NaN,
    "5. Tiền thu từ bán, thanh lý bất động sản đầu tư": NaN,
    "6. Tiền chi ra do bán, thanh lý bất động sản đầu tư": NaN,
    "7. Tiền chi đầu tư, góp vốn vào các đơn vị khác (mua công ty con, góp vốn liên doanh, liên kết, đầu tư dài hạn khác)": "-60167000000",
    "8. Tiền thu đầu tư, góp vốn vào các đơn vị khác (bán, thanh lý Công ty con, góp vốn liên doanh,  liên kết, đầu tư dài hạn khác)": NaN,
    "9. Tiền thu cổ tức và lợi nhuận được chia từ các khoản đầu tư, góp vốn dài hạn": "118576000000",
    "II- Lưu chuyển tiền thuần từ hoạt động đầu tư": "-1379136000000",
    "Lưu chuyển tiền từ hoạt động tài chính": NaN,
    "2. Tiền thu từ phát hành giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "3. Tiền chi thanh toán giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "4. Cổ tức trả cho cổ đông, lợi nhuận đã chia": "-3776798000000",
    "5. Tiền chi ra mua cổ phiếu quỹ": NaN,
    "6. Tiền thu được do bán cổ phiếu quỹ": NaN,
    "III- Lưu chuyển tiền thuần từ hoạt động tài chính": "-3776798000000",
    "IV- Lưu chuyển tiền thuần trong kỳ": "110185283000000",
    "V- Tiền và các khoản tương đương tiền tại thời điểm đầu kỳ": "430614185000000",
    "VI- Điều chỉnh ảnh hưởng của thay đổi tỷ giá": NaN,
    "VII. Tiền và các khoản tương đương tiền tại thời điểm cuối kỳ": "540799468000000",
    "1. Tăng v ốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu": NaN
  },
  {
    "period": 2024,
    "year_period": 2024,
    "Lưu chuyển tiền từ hoạt động kinh doanh": NaN,
    "1. Thu nhập lãi và các khoản thu nhập tương tự nhận được": "93772270000000",
    "2. Chi phí lãi và các chi phí tương tự đã trả": "-43790244000000",
    "3. Thu nhập từ hoạt động dịch vụ nhận được": "3523997000000",
    "4- Chênh lệch số tiền thực thu/thực chi từ hoạt động kinh doanh (ngoại tệ, vàng bạc, chứng khoán)": "4094518000000",
    "5. Thu nhập khác": "-1390558000000",
    "6. Tiền thu các khoản nợ đã được xử lý xóa, bù đắp bằng nguồn rủi ro": "3751009000000",
    "7. Tiền chi trả cho nhân viên và hoạt động quản lý, công vụ": "-20922713000000",
    "8. Tiền thuế thu nhập thực nộp trong kỳ": "-8854401000000",
    "Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động": "30183878000000",
    "Những thay đổi về tài sản hoạt động": NaN,
    "9. (Tăng)/Giảm các khoản tiền, vàng gửi và cho vay các TCTD khác": "18021934000000",
    "10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán": "-24134346000000",
    "11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác": "-1314434000000",
    "12. (Tăng)/Giảm các khoản cho vay khách hàng": "-178839881000000",
    "13. Giảm nguồn dự phòng để bù đắp tổn thất các khoản": "-5358023000000",
    "14. (Tăng)/Giảm khác về tài sản hoạt động": "496225000000",
    "Những thay đổi về công nợ hoạt động": NaN,
    "15. Tăng/(Giảm) các khoản nợ chính phủ và NHNN": "76566500000000",
    "16. Tăng/(Giảm) các khoản tiền gửi, tiền vay các tổ chức tín dụng": "20694977000000",
    "17. Tăng/(Giảm) tiền gửi của khách hàng (bao gồm cả Kho bạc Nhà nước)": "118967239000000",
    "18. Tăng/(Giảm) phát hành giấy tờ có giá (ngoại trừ giấy tờ có giá phát hành được tình vào hoạt động tài chính)": "4212146000000",
    "19. Tăng/(Giảm) vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": "164000000",
    "20. Tăng/(Giảm) các công cụ tài chính phái sinh và các khoản nợ tài chính khác": "-117752000000",
    "21. Tăng/(Giảm) khác về công nợ hoạt động": "2620930000000",
    "22. Chi từ các quỹ của TCTD": "-2876726000000",
    "I - Lưu chuyển tiền thuần từ hoạt động kinh doanh": "59122831000000",
    "Lưu chuyển tiền từ hoạt động đầu tư": NaN,
    "1. Mua sắm tài sản cố định": "-1480121000000",
    "2. Tiền thu từ thanh lý, nhượng bán TSCĐ": "12504000000",
    "3. Tiền chi từ thanh lý, nhượng bán TSCĐ": "-1252000000",
    "4. Mua sắm bất động sản đầu tư": NaN,
    "5. Tiền thu từ bán, thanh lý bất động sản đầu tư": NaN,
    "6. Tiền chi ra do bán, thanh lý bất động sản đầu tư": NaN,
    "7. Tiền chi đầu tư, góp vốn vào các đơn vị khác (mua công ty con, góp vốn liên doanh, liên kết, đầu tư dài hạn khác)": NaN,
    "8. Tiền thu đầu tư, góp vốn vào các đơn vị khác (bán, thanh lý Công ty con, góp vốn liên doanh,  liên kết, đầu tư dài hạn khác)": "747000000",
    "9. Tiền thu cổ tức và lợi nhuận được chia từ các khoản đầu tư, góp vốn dài hạn": "160709000000",
    "II- Lưu chuyển tiền thuần từ hoạt động đầu tư": "-1307413000000",
    "Lưu chuyển tiền từ hoạt động tài chính": NaN,
    "2. Tiền thu từ phát hành giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "3. Tiền chi thanh toán giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "4. Cổ tức trả cho cổ đông, lợi nhuận đã chia": "-19963000000",
    "5. Tiền chi ra mua cổ phiếu quỹ": NaN,
    "6. Tiền thu được do bán cổ phiếu quỹ": NaN,
    "III- Lưu chuyển tiền thuần từ hoạt động tài chính": "-19963000000",
    "IV- Lưu chuyển tiền thuần trong kỳ": "57795455000000",
    "V- Tiền và các khoản tương đương tiền tại thời điểm đầu kỳ": "372818730000000",
    "VI- Điều chỉnh ảnh hưởng của thay đổi tỷ giá": NaN,
    "VII. Tiền và các khoản tương đương tiền tại thời điểm cuối kỳ": "430614185000000",
    "1. Tăng v ốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu": NaN
  },
  {
    "period": 2023,
    "year_period": 2023,
    "Lưu chuyển tiền từ hoạt động kinh doanh": NaN,
    "1. Thu nhập lãi và các khoản thu nhập tương tự nhận được": "108115649000000",
    "2. Chi phí lãi và các chi phí tương tự đã trả": "-47454819000000",
    "3. Thu nhập từ hoạt động dịch vụ nhận được": "4100623000000",
    "4- Chênh lệch số tiền thực thu/thực chi từ hoạt động kinh doanh (ngoại tệ, vàng bạc, chứng khoán)": "5242957000000",
    "5. Thu nhập khác": "179388000000",
    "6. Tiền thu các khoản nợ đã được xử lý xóa, bù đắp bằng nguồn rủi ro": "2090116000000",
    "7. Tiền chi trả cho nhân viên và hoạt động quản lý, công vụ": "-19932948000000",
    "8. Tiền thuế thu nhập thực nộp trong kỳ": "-8969967000000",
    "Lưu chuyển tiền thuần từ hoạt động kinh doanh trước những thay đổi về tài sản và vốn lưu động": "43370999000000",
    "Những thay đổi về tài sản hoạt động": NaN,
    "9. (Tăng)/Giảm các khoản tiền, vàng gửi và cho vay các TCTD khác": "10780289000000",
    "10. (Tăng)/Giảm các khoản về kinh doanh chứng khoán": "19792999000000",
    "11. (Tăng)/Giảm các công cụ tài chính phái sinh và các công cụ tài chính khác": "156515000000",
    "12. (Tăng)/Giảm các khoản cho vay khách hàng": "-125292768000000",
    "13. Giảm nguồn dự phòng để bù đắp tổn thất các khoản": "-5758202000000",
    "14. (Tăng)/Giảm khác về tài sản hoạt động": "30781829000000",
    "Những thay đổi về công nợ hoạt động": NaN,
    "15. Tăng/(Giảm) các khoản nợ chính phủ và NHNN": "-65643979000000",
    "16. Tăng/(Giảm) các khoản tiền gửi, tiền vay các tổ chức tín dụng": "-18671870000000",
    "17. Tăng/(Giảm) tiền gửi của khách hàng (bao gồm cả Kho bạc Nhà nước)": "152229140000000",
    "18. Tăng/(Giảm) phát hành giấy tờ có giá (ngoại trừ giấy tờ có giá phát hành được tình vào hoạt động tài chính)": "-5425274000000",
    "19. Tăng/(Giảm) vốn tài trợ, ủy thác đầu tư, cho vay mà TCTD chịu rủi ro": "-2933000000",
    "20. Tăng/(Giảm) các công cụ tài chính phái sinh và các khoản nợ tài chính khác": "117752000000",
    "21. Tăng/(Giảm) khác về công nợ hoạt động": "-72173193000000",
    "22. Chi từ các quỹ của TCTD": "-2802834000000",
    "I - Lưu chuyển tiền thuần từ hoạt động kinh doanh": "-38541530000000",
    "Lưu chuyển tiền từ hoạt động đầu tư": NaN,
    "1. Mua sắm tài sản cố định": "-1008160000000",
    "2. Tiền thu từ thanh lý, nhượng bán TSCĐ": "9435000000",
    "3. Tiền chi từ thanh lý, nhượng bán TSCĐ": "-6770000000",
    "4. Mua sắm bất động sản đầu tư": NaN,
    "5. Tiền thu từ bán, thanh lý bất động sản đầu tư": NaN,
    "6. Tiền chi ra do bán, thanh lý bất động sản đầu tư": NaN,
    "7. Tiền chi đầu tư, góp vốn vào các đơn vị khác (mua công ty con, góp vốn liên doanh, liên kết, đầu tư dài hạn khác)": NaN,
    "8. Tiền thu đầu tư, góp vốn vào các đơn vị khác (bán, thanh lý Công ty con, góp vốn liên doanh,  liên kết, đầu tư dài hạn khác)": NaN,
    "9. Tiền thu cổ tức và lợi nhuận được chia từ các khoản đầu tư, góp vốn dài hạn": "146088000000",
    "II- Lưu chuyển tiền thuần từ hoạt động đầu tư": "-859407000000",
    "Lưu chuyển tiền từ hoạt động tài chính": NaN,
    "2. Tiền thu từ phát hành giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "3. Tiền chi thanh toán giấy tờ có giá dài hạn có đủ điều kiện tính vào vốn tự có và các khoản vốn vay dài hạn khác": NaN,
    "4. Cổ tức trả cho cổ đông, lợi nhuận đã chia": "-15627000000",
    "5. Tiền chi ra mua cổ phiếu quỹ": NaN,
    "6. Tiền thu được do bán cổ phiếu quỹ": NaN,
    "III- Lưu chuyển tiền thuần từ hoạt động tài chính": "-15627000000",
    "IV- Lưu chuyển tiền thuần trong kỳ": "-39416564000000",
    "V- Tiền và các khoản tương đương tiền tại thời điểm đầu kỳ": "412235294000000",
    "VI- Điều chỉnh ảnh hưởng của thay đổi tỷ giá": NaN,
    "VII. Tiền và các khoản tương đương tiền tại thời điểm cuối kỳ": "372818730000000",
    "1. Tăng v ốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu": NaN
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.financial.Finance`
- Provider method: `cash_flow`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:18:57.032181+00:00`
- Success: `True`
- Row count: `8`

```text
report_period, ticker, Operating profit/(loss) before changes in Working Capital, Corporate income tax paid, Net cash from operating activities, Purchases of fixed assets and other long term assets, Proceeds from disposal of fixed assets, Investments in other entities, Proceeds from divestment in other entities, Dividends and interest received, Net cash from investing activities, Dividends paid, Net Increase/(Decrease) in cash and cash equivalents, Cash and cash equivalents at the beginning of period, Effect of foreign exchange differences, Cash and cash equivalents at end of the period, Payments for corporate income tax, cfb46, cfb47, (Increase)/decrease in compulsory reserves with the SBV, Increase/(decrease) in placements with and loans to other credit institutions, Increase/(decrease) in trading securities, Increase/(decrease) in derivatives and other financial assets, Increase/(decrease) in loans and advances to customers, (Increase)/decrease in interest receivable, Increase/(decrease) in other operating assets, Increase/(decrease) in Loans from the State and SBV, Increase/(decrease) in placements and Loans from other credit institutions, Increase/(decrease) in deposits from customers, Increase/(decrease) in derivatives and other financial liabilities, Increase/(decrease) in funds received from Gov, international and other institutions, Increase/(decrease) in accrued interest expenses, Increase/(decrease) in other operating liabilities, Net cash flows from operating activities before CIT, Payment from reserves, Bad debt recoveries, Payments on disposal of fixed assets, Purchases of investment properties, Proceeds from disposal of investment properties, Payments on disposal of investment properties, Payments for redemption of convertible bonds, Purchase of treasury shares, Proceeds from selling of treasury shares, Interest and similar receipts, Interest and similar payments, Net receipts from dealing of foreign currencies, gold, Net receipts from dealing of securities, Other operating income, Payments to employees and other operating expenses, Receipts from debts written off or paid off by risk fund, Net receipts from foreign currencies, gold and securities trading
```
- Dtypes: `{'report_period': 'str', 'ticker': 'str', 'Operating profit/(loss) before changes in Working Capital': 'float64', 'Corporate income tax paid': 'float64', 'Net cash from operating activities': 'float64', 'Purchases of fixed assets and other long term assets': 'float64', 'Proceeds from disposal of fixed assets': 'float64', 'Investments in other entities': 'float64', 'Proceeds from divestment in other entities': 'float64', 'Dividends and interest received': 'float64', 'Net cash from investing activities': 'float64', 'Dividends paid': 'float64', 'Net Increase/(Decrease) in cash and cash equivalents': 'float64', 'Cash and cash equivalents at the beginning of period': 'float64', 'Effect of foreign exchange differences': 'float64', 'Cash and cash equivalents at end of the period': 'float64', 'Payments for corporate income tax': 'float64', 'cfb46': 'float64', 'cfb47': 'float64', '(Increase)/decrease in compulsory reserves with the SBV': 'float64', 'Increase/(decrease) in placements with and loans to other credit institutions': 'float64', 'Increase/(decrease) in trading securities': 'float64', 'Increase/(decrease) in derivatives and other financial assets': 'float64', 'Increase/(decrease) in loans and advances to customers': 'float64', '(Increase)/decrease in interest receivable': 'float64', 'Increase/(decrease) in other operating assets': 'float64', 'Increase/(decrease) in Loans from the State and SBV': 'float64', 'Increase/(decrease) in placements and Loans from other credit institutions': 'float64', 'Increase/(decrease) in deposits from customers': 'float64', 'Increase/(decrease) in derivatives and other financial liabilities': 'float64', 'Increase/(decrease) in funds received from Gov, international and other institutions': 'float64', 'Increase/(decrease) in accrued interest expenses': 'float64', 'Increase/(decrease) in other operating liabilities': 'float64', 'Net cash flows from operating activities before CIT': 'float64', 'Payment from reserves': 'float64', 'Bad debt recoveries': 'float64', 'Payments on disposal of fixed assets': 'float64', 'Purchases of investment properties': 'float64', 'Proceeds from disposal of investment properties': 'float64', 'Payments on disposal of investment properties': 'float64', 'Payments for redemption of convertible bonds': 'float64', 'Purchase of treasury shares': 'float64', 'Proceeds from selling of treasury shares': 'float64', 'Interest and similar receipts': 'float64', 'Interest and similar payments': 'float64', 'Net receipts from dealing of foreign currencies, gold': 'float64', 'Net receipts from dealing of securities': 'float64', 'Other operating income': 'float64', 'Payments to employees and other operating expenses': 'float64', 'Receipts from debts written off or paid off by risk fund': 'float64', 'Net receipts from foreign currencies, gold and securities trading': 'float64'}`

```json
[
  {
    "report_period": "year",
    "ticker": "VCB",
    "Operating profit/(loss) before changes in Working Capital": 21540011000000.0,
    "Corporate income tax paid": 0.0,
    "Net cash from operating activities": -60234337000000.0,
    "Purchases of fixed assets and other long term assets": -1133639000000.0,
    "Proceeds from disposal of fixed assets": 6767000000.0,
    "Investments in other entities": 0.0,
    "Proceeds from divestment in other entities": 2628038000000.0,
    "Dividends and interest received": 129753000000.0,
    "Net cash from investing activities": 1629227000000.0,
    "Dividends paid": -2914981000000.0,
    "Net Increase/(Decrease) in cash and cash equivalents": -2914981000000.0,
    "Cash and cash equivalents at the beginning of period": 305534247000000.0,
    "Effect of foreign exchange differences": 0.0,
    "Cash and cash equivalents at end of the period": 244014156000000.0,
    "Payments for corporate income tax": -2585774000000.0,
    "cfb46": 0.0,
    "cfb47": 0.0,
    "(Increase)/decrease in compulsory reserves with the SBV": 0.0,
    "Increase/(decrease) in placements with and loans to other credit institutions": 2306008000000.0,
    "Increase/(decrease) in trading securities": -14454777000000.0,
    "Increase/(decrease) in derivatives and other financial assets": 556371000000.0,
    "Increase/(decrease) in loans and advances to customers": -88432298000000.0,
    "(Increase)/decrease in interest receivable": 0.0,
    "Increase/(decrease) in other operating assets": -2666699000000.0,
    "Increase/(decrease) in Loans from the State and SBV": -80699753000000.0,
    "Increase/(decrease) in placements and Loans from other credit institutions": 9581876000000.0,
    "Increase/(decrease) in deposits from customers": 93409398000000.0,
    "Increase/(decrease) in derivatives and other financial liabilities": 0.0,
    "Increase/(decrease) in funds received from Gov, international and other institutions": 2650000000.0,
    "Increase/(decrease) in accrued interest expenses": 0.0,
    "Increase/(decrease) in other operating liabilities": 824980000000.0,
    "Net cash flows from operating activities before CIT": -58869162000000.0,
    "Payment from reserves": -1365175000000.0,
    "Bad debt recoveries": 0.0,
    "Payments on disposal of fixed assets": -1692000000.0,
    "Purchases of investment properties": 0.0,
    "Proceeds from disposal of investment properties": 0.0,
    "Payments on disposal of investment properties": 0.0,
    "Payments for redemption of convertible bonds": 0.0,
    "Purchase of treasury shares": 0.0,
    "Proceeds from selling of treasury shares": 0.0,
    "Interest and similar receipts": 54473260000000.0,
    "Interest and similar payments": -27395363000000.0,
    "Net receipts from dealing of foreign currencies, gold": 3203390000000.0,
    "Net receipts from dealing of securities": 0.0,
    "Other operating income": -42957000000.0,
    "Payments to employees and other operating expenses": -12787284000000.0,
    "Receipts from debts written off or paid off by risk fund": 3272247000000.0,
    "Net receipts from foreign currencies, gold and securities trading": 3203390000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Operating profit/(loss) before changes in Working Capital": 26469418000000.0,
    "Corporate income tax paid": 0.0,
    "Net cash from operating activities": 25075832000000.0,
    "Purchases of fixed assets and other long term assets": -1005065000000.0,
    "Proceeds from disposal of fixed assets": 11589000000.0,
    "Investments in other entities": 0.0,
    "Proceeds from divestment in other entities": 95773000000.0,
    "Dividends and interest received": 197571000000.0,
    "Net cash from investing activities": -706447000000.0,
    "Dividends paid": -2219483000000.0,
    "Net Increase/(Decrease) in cash and cash equivalents": 3886994000000.0,
    "Cash and cash equivalents at the beginning of period": 244014156000000.0,
    "Effect of foreign exchange differences": 0.0,
    "Cash and cash equivalents at end of the period": 272270535000000.0,
    "Payments for corporate income tax": -4827328000000.0,
    "cfb46": 0.0,
    "cfb47": 0.0,
    "(Increase)/decrease in compulsory reserves with the SBV": 0.0,
    "Increase/(decrease) in placements with and loans to other credit institutions": 2189341000000.0,
    "Increase/(decrease) in trading securities": -17562680000000.0,
    "Increase/(decrease) in derivatives and other financial assets": 177671000000.0,
    "Increase/(decrease) in loans and advances to customers": -102840133000000.0,
    "(Increase)/decrease in interest receivable": 0.0,
    "Increase/(decrease) in other operating assets": -3354149000000.0,
    "Increase/(decrease) in Loans from the State and SBV": 1680491000000.0,
    "Increase/(decrease) in placements and Loans from other credit institutions": -2906994000000.0,
    "Increase/(decrease) in deposits from customers": 126521754000000.0,
    "Increase/(decrease) in derivatives and other financial liabilities": 0.0,
    "Increase/(decrease) in funds received from Gov, international and other institutions": -5372000000.0,
    "Increase/(decrease) in accrued interest expenses": 0.0,
    "Increase/(decrease) in other operating liabilities": 720675000000.0,
    "Net cash flows from operating activities before CIT": 26510053000000.0,
    "Payment from reserves": -1434221000000.0,
    "Bad debt recoveries": 0.0,
    "Payments on disposal of fixed assets": -6315000000.0,
    "Purchases of investment properties": 0.0,
    "Proceeds from disposal of investment properties": 0.0,
    "Payments on disposal of investment properties": 0.0,
    "Payments for redemption of convertible bonds": 0.0,
    "Purchase of treasury shares": 0.0,
    "Proceeds from selling of treasury shares": 0.0,
    "Interest and similar receipts": 66956606000000.0,
    "Interest and similar payments": -31567168000000.0,
    "Net receipts from dealing of foreign currencies, gold": 0.0,
    "Net receipts from dealing of securities": 3068655000000.0,
    "Other operating income": -115005000000.0,
    "Payments to employees and other operating expenses": -14532712000000.0,
    "Receipts from debts written off or paid off by risk fund": 3179526000000.0,
    "Net receipts from foreign currencies, gold and securities trading": 3068655000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Operating profit/(loss) before changes in Working Capital": 29528536000000.0,
    "Corporate income tax paid": 0.0,
    "Net cash from operating activities": 25603435000000.0,
    "Purchases of fixed assets and other long term assets": -3001902000000.0,
    "Proceeds from disposal of fixed assets": 10569000000.0,
    "Investments in other entities": 0.0,
    "Proceeds from divestment in other entities": 605274000000.0,
    "Dividends and interest received": 83959000000.0,
    "Net cash from investing activities": -2305153000000.0,
    "Dividends paid": -2986115000000.0,
    "Net Increase/(Decrease) in cash and cash equivalents": -2986115000000.0,
    "Cash and cash equivalents at the beginning of period": 272270535000000.0,
    "Effect of foreign exchange differences": 0.0,
    "Cash and cash equivalents at end of the period": 292582702000000.0,
    "Payments for corporate income tax": -4680317000000.0,
    "cfb46": 0.0,
    "cfb47": 0.0,
    "(Increase)/decrease in compulsory reserves with the SBV": 0.0,
    "Increase/(decrease) in placements with and loans to other credit institutions": 4040576000000.0,
    "Increase/(decrease) in trading securities": 10336784000000.0,
    "Increase/(decrease) in derivatives and other financial assets": 98312000000.0,
    "Increase/(decrease) in loans and advances to customers": -105081370000000.0,
    "(Increase)/decrease in interest receivable": 0.0,
    "Increase/(decrease) in other operating assets": 1613500000000.0,
    "Increase/(decrease) in Loans from the State and SBV": -51188811000000.0,
    "Increase/(decrease) in placements and Loans from other credit institutions": 29966748000000.0,
    "Increase/(decrease) in deposits from customers": 103662698000000.0,
    "Increase/(decrease) in derivatives and other financial liabilities": 52031000000.0,
    "Increase/(decrease) in funds received from Gov, international and other institutions": -5752000000.0,
    "Increase/(decrease) in accrued interest expenses": 0.0,
    "Increase/(decrease) in other operating liabilities": 7396305000000.0,
    "Net cash flows from operating activities before CIT": 27469577000000.0,
    "Payment from reserves": -1866142000000.0,
    "Bad debt recoveries": 0.0,
    "Payments on disposal of fixed assets": -3053000000.0,
    "Purchases of investment properties": 0.0,
    "Proceeds from disposal of investment properties": 0.0,
    "Payments on disposal of investment properties": 0.0,
    "Payments for redemption of convertible bonds": 0.0,
    "Purchase of treasury shares": 0.0,
    "Proceeds from selling of treasury shares": 0.0,
    "Interest and similar receipts": 70063617000000.0,
    "Interest and similar payments": -33194191000000.0,
    "Net receipts from dealing of foreign currencies, gold": 0.0,
    "Net receipts from dealing of securities": 3464399000000.0,
    "Other operating income": -628988000000.0,
    "Payments to employees and other operating expenses": -14525026000000.0,
    "Receipts from debts written off or paid off by risk fund": 2421725000000.0,
    "Net receipts from foreign currencies, gold and securities trading": 3464399000000.0
  }
]
```

#### Notes / caveats

Retrieve cash flow statement data.

### history

- Kind: `method`
- Signature: `(*A, **B)`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `A` | `VAR_POSITIONAL` | `True` | `None` | `` |
| `B` | `VAR_KEYWORD` | `True` | `None` | `` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `mas`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### income_statement

- Kind: `method`
- Signature: `(period = None, limit = 12, include_metadata = False, display_mode = "<FieldDisplayMode.STD: 'std'>", show_log = False) -> DataFrame chứa báo cáo kết quả kinh doanh.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa báo cáo kết quả kinh doanh.`
- Purpose: Retrieve income statement data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `12` | `` |  | `5` |  | Số kỳ báo cáo tối đa cần lấy. Mặc định 4. |
| `include_metadata` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `omitted; default False` |  | Bao gồm thông tin audit và unit trong rows. Mặc định False. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.financial.Finance`
- Provider method: `income_statement`

```text
item, item_id, unit, periods
```
- Note: Derived from provider docstring column hints.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:09.945057+00:00`
- Success: `True`
- Row count: `24`

```text
item, item_id, 2025, 2024, 2023, 2022, 2021
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64', '2021': 'float64'}`

```json
[
  {
    "item": "1. Thu nhập lãi và các khoản thu nhập tương tự",
    "item_id": "1_interest_income_and_similar_income",
    "2025": 105119449000.0,
    "2024": 93654841000.0,
    "2023": 108122278000.0,
    "2022": 88112700000.0,
    "2021": 70621957000.0
  },
  {
    "item": "2. Chi phí lãi và các chi phí tương tự",
    "item_id": "2_interest_expense_and_similar_expenses",
    "2025": 46445074000.0,
    "2024": 38249106000.0,
    "2023": 54501409000.0,
    "2022": 34866222000.0,
    "2021": 28349385000.0
  },
  {
    "item": "3. Thu nhập từ hoạt động dịch vụ",
    "item_id": "3_fee_and_commission_income",
    "2025": 11854531000.0,
    "2024": 13143005000.0,
    "2023": 12632739000.0,
    "2022": 12425007000.0,
    "2021": 11286516000.0
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.financial.Finance`
- Provider method: `income_statement`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:11.373089+00:00`
- Success: `True`
- Row count: `14`

```text
period, year_period, 1. Thu nhập lãi và các khoản thu nhập tương tự, 2. Chi phí lãi và các chi phí tương tự, I. Thu nhập lãi thuần, 3. Thu nhập từ hoạt động dịch vụ, 4. Chi phí hoạt động dịch vụ, II. Lãi/lỗ thuần từ hoạt động dịch vụ, III. Lãi/lỗ thuần từ hoạt động kinh doanh ngoại hối và vàng, IV. Lãi/lỗ thuần từ mua bán chứng khoán kinh doanh, V. Lãi/lỗ thuần từ mua bán chứng khoán đầu tư, 5. Thu nhập từ hoạt động khác, 6. Chi phí hoạt động khác, VI. Lãi/lỗ thuần từ hoạt động khác, VII. Thu nhập từ góp vốn, mua cổ phần, VIII. Chi phí hoạt động, IX. Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng (I+II+III+IV+V+VI+VII-VIII), X. Chi phí dự phòng rủi ro tín dụng, XI. Tổng lợi nhuận trước thuế (IX-X), 7. Chi phí thuế TNDN hiện hành, 8. Chi phí thuế TNDN hoãn lại, XII. Chi phí thuế TNDN, XIII. Lợi nhuận sau thuế (XI-XII), XIV. Lợi ích của cổ đông thiểu số, XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV), Lãi cơ bản trên cổ phiếu (BCTC) (VNÐ)
```
- Dtypes: `{'period': 'int64', 'year_period': 'int64', '1. Thu nhập lãi và các khoản thu nhập tương tự': 'str', '2. Chi phí lãi và các chi phí tương tự': 'str', 'I. Thu nhập lãi thuần': 'str', '3. Thu nhập từ hoạt động dịch vụ': 'str', '4. Chi phí hoạt động dịch vụ': 'str', 'II. Lãi/lỗ thuần từ hoạt động dịch vụ': 'str', 'III. Lãi/lỗ thuần từ hoạt động kinh doanh ngoại hối và vàng': 'str', 'IV. Lãi/lỗ thuần từ mua bán chứng khoán kinh doanh': 'str', 'V. Lãi/lỗ thuần từ mua bán chứng khoán đầu tư': 'str', '5. Thu nhập từ hoạt động khác': 'str', '6. Chi phí hoạt động khác': 'str', 'VI. Lãi/lỗ thuần từ hoạt động khác': 'str', 'VII. Thu nhập từ góp vốn, mua cổ phần': 'str', 'VIII. Chi phí hoạt động': 'str', 'IX. Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng (I+II+III+IV+V+VI+VII-VIII)': 'str', 'X. Chi phí dự phòng rủi ro tín dụng': 'str', 'XI. Tổng lợi nhuận trước thuế (IX-X)': 'str', '7. Chi phí thuế TNDN hiện hành': 'str', '8. Chi phí thuế TNDN hoãn lại': 'str', 'XII. Chi phí thuế TNDN': 'str', 'XIII. Lợi nhuận sau thuế (XI-XII)': 'str', 'XIV. Lợi ích của cổ đông thiểu số': 'str', 'XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)': 'str', 'Lãi cơ bản trên cổ phiếu (BCTC) (VNÐ)': 'str'}`

```json
[
  {
    "period": 2025,
    "year_period": 2025,
    "1. Thu nhập lãi và các khoản thu nhập tương tự": "105119449000000",
    "2. Chi phí lãi và các chi phí tương tự": "46445074000000",
    "I. Thu nhập lãi thuần": "58674375000000",
    "3. Thu nhập từ hoạt động dịch vụ": "11854531000000",
    "4. Chi phí hoạt động dịch vụ": "8384664000000",
    "II. Lãi/lỗ thuần từ hoạt động dịch vụ": "3469867000000",
    "III. Lãi/lỗ thuần từ hoạt động kinh doanh ngoại hối và vàng": "6165112000000",
    "IV. Lãi/lỗ thuần từ mua bán chứng khoán kinh doanh": "171160000000",
    "V. Lãi/lỗ thuần từ mua bán chứng khoán đầu tư": "3616000000",
    "5. Thu nhập từ hoạt động khác": "5269106000000",
    "6. Chi phí hoạt động khác": "1677513000000",
    "VI. Lãi/lỗ thuần từ hoạt động khác": "3591593000000",
    "VII. Thu nhập từ góp vốn, mua cổ phần": "281863000000",
    "VIII. Chi phí hoạt động": "25152290000000",
    "IX. Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng (I+II+III+IV+V+VI+VII-VIII)": "47205296000000",
    "X. Chi phí dự phòng rủi ro tín dụng": "3185040000000",
    "XI. Tổng lợi nhuận trước thuế (IX-X)": "44020256000000",
    "7. Chi phí thuế TNDN hiện hành": "7843123000000",
    "8. Chi phí thuế TNDN hoãn lại": "978700000000",
    "XII. Chi phí thuế TNDN": "8821823000000",
    "XIII. Lợi nhuận sau thuế (XI-XII)": "35198433000000",
    "XIV. Lợi ích của cổ đông thiểu số": "20278000000",
    "XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)": "35178155000000",
    "Lãi cơ bản trên cổ phiếu (BCTC) (VNÐ)": "4210"
  },
  {
    "period": 2024,
    "year_period": 2024,
    "1. Thu nhập lãi và các khoản thu nhập tương tự": "93654841000000",
    "2. Chi phí lãi và các chi phí tương tự": "38249106000000",
    "I. Thu nhập lãi thuần": "55405735000000",
    "3. Thu nhập từ hoạt động dịch vụ": "13143005000000",
    "4. Chi phí hoạt động dịch vụ": "8006444000000",
    "II. Lãi/lỗ thuần từ hoạt động dịch vụ": "5136561000000",
    "III. Lãi/lỗ thuần từ hoạt động kinh doanh ngoại hối và vàng": "5291751000000",
    "IV. Lãi/lỗ thuần từ mua bán chứng khoán kinh doanh": "62123000000",
    "V. Lãi/lỗ thuần từ mua bán chứng khoán đầu tư": "3444000000",
    "5. Thu nhập từ hoạt động khác": "4468806000000",
    "6. Chi phí hoạt động khác": "2097103000000",
    "VI. Lãi/lỗ thuần từ hoạt động khác": "2371703000000",
    "VII. Thu nhập từ góp vốn, mua cổ phần": "307179000000",
    "VIII. Chi phí hoạt động": "23027363000000",
    "IX. Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng (I+II+III+IV+V+VI+VII-VIII)": "45551133000000",
    "X. Chi phí dự phòng rủi ro tín dụng": "3314998000000",
    "XI. Tổng lợi nhuận trước thuế (IX-X)": "42236135000000",
    "7. Chi phí thuế TNDN hiện hành": "8526496000000",
    "8. Chi phí thuế TNDN hoãn lại": "-143478000000",
    "XII. Chi phí thuế TNDN": "8383018000000",
    "XIII. Lợi nhuận sau thuế (XI-XII)": "33853117000000",
    "XIV. Lợi ích của cổ đông thiểu số": "21731000000",
    "XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)": "33831386000000",
    "Lãi cơ bản trên cổ phiếu (BCTC) (VNÐ)": "5571"
  },
  {
    "period": 2023,
    "year_period": 2023,
    "1. Thu nhập lãi và các khoản thu nhập tương tự": "108122278000000",
    "2. Chi phí lãi và các chi phí tương tự": "54501409000000",
    "I. Thu nhập lãi thuần": "53620869000000",
    "3. Thu nhập từ hoạt động dịch vụ": "12632739000000",
    "4. Chi phí hoạt động dịch vụ": "6853016000000",
    "II. Lãi/lỗ thuần từ hoạt động dịch vụ": "5779723000000",
    "III. Lãi/lỗ thuần từ hoạt động kinh doanh ngoại hối và vàng": "5660028000000",
    "IV. Lãi/lỗ thuần từ mua bán chứng khoán kinh doanh": "124217000000",
    "V. Lãi/lỗ thuần từ mua bán chứng khoán đầu tư": NaN,
    "5. Thu nhập từ hoạt động khác": "4050144000000",
    "6. Chi phí hoạt động khác": "1777975000000",
    "VI. Lãi/lỗ thuần từ hoạt động khác": "2272169000000",
    "VII. Thu nhập từ góp vốn, mua cổ phần": "266456000000",
    "VIII. Chi phí hoạt động": "21914899000000",
    "IX. Lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng (I+II+III+IV+V+VI+VII-VIII)": "45808563000000",
    "X. Chi phí dự phòng rủi ro tín dụng": "4564876000000",
    "XI. Tổng lợi nhuận trước thuế (IX-X)": "41243687000000",
    "7. Chi phí thuế TNDN hiện hành": "8079401000000",
    "8. Chi phí thuế TNDN hoãn lại": "109838000000",
    "XII. Chi phí thuế TNDN": "8189239000000",
    "XIII. Lợi nhuận sau thuế (XI-XII)": "33054448000000",
    "XIV. Lợi ích của cổ đông thiểu số": "21245000000",
    "XV. Lợi nhuận sau thuế của cổ đông của Ngân hàng mẹ (XIII-XIV)": "33033203000000",
    "Lãi cơ bản trên cổ phiếu (BCTC) (VNÐ)": "5449"
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.financial.Finance`
- Provider method: `income_statement`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:12.237087+00:00`
- Success: `True`
- Row count: `8`

```text
report_period, ticker, Net Accounting Profit/(loss) before tax, Business income tax - current, Business income tax - deferred, Business income tax expenses, Net profit/(loss) after tax, Minority interest, Attributable to parent company, EPS basic (VND), EPS diluted (VND), Interest and Similar Income, Interest and Similar Expenses, Net Interest Income, Net gain/(loss) from foreign currency and gold dealings, Net gain/(loss) from trading of trading securities, Net gain/(loss) from disposal of investment securities, Other Income, Other Expenses, Net Other income/expenses, Dividends Income, Total Operating Income, General and Admin Expenses, Net Operating Profit Before Allowance for Credit Loss
```
- Dtypes: `{'report_period': 'str', 'ticker': 'str', 'Net Accounting Profit/(loss) before tax': 'float64', 'Business income tax - current': 'float64', 'Business income tax - deferred': 'float64', 'Business income tax expenses': 'float64', 'Net profit/(loss) after tax': 'float64', 'Minority interest': 'float64', 'Attributable to parent company': 'float64', 'EPS basic (VND)': 'float64', 'EPS diluted (VND)': 'float64', 'Interest and Similar Income': 'float64', 'Interest and Similar Expenses': 'float64', 'Net Interest Income': 'float64', 'Net gain/(loss) from foreign currency and gold dealings': 'float64', 'Net gain/(loss) from trading of trading securities': 'float64', 'Net gain/(loss) from disposal of investment securities': 'float64', 'Other Income': 'float64', 'Other Expenses': 'float64', 'Net Other income/expenses': 'float64', 'Dividends Income': 'float64', 'Total Operating Income': 'float64', 'General and Admin Expenses': 'float64', 'Net Operating Profit Before Allowance for Credit Loss': 'float64'}`

```json
[
  {
    "report_period": "year",
    "ticker": "VCB",
    "Net Accounting Profit/(loss) before tax": 18269226000000.0,
    "Business income tax - current": -3648356000000.0,
    "Business income tax - deferred": 1192000000.0,
    "Business income tax expenses": -3647164000000.0,
    "Net profit/(loss) after tax": 14622062000000.0,
    "Minority interest": -16484000000.0,
    "Attributable to parent company": 14605578000000.0,
    "EPS basic (VND)": 3323.0,
    "EPS diluted (VND)": 0.0,
    "Interest and Similar Income": 55863951000000.0,
    "Interest and Similar Expenses": -27455435000000.0,
    "Net Interest Income": 28408516000000.0,
    "Net gain/(loss) from foreign currency and gold dealings": 2266429000000.0,
    "Net gain/(loss) from trading of trading securities": 250462000000.0,
    "Net gain/(loss) from disposal of investment securities": 0.0,
    "Other Income": 3515904000000.0,
    "Other Expenses": -281539000000.0,
    "Net Other income/expenses": 3234365000000.0,
    "Dividends Income": 1716169000000.0,
    "Total Operating Income": 39278433000000.0,
    "General and Admin Expenses": -13611094000000.0,
    "Net Operating Profit Before Allowance for Credit Loss": 25667339000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Net Accounting Profit/(loss) before tax": 23211571000000.0,
    "Business income tax - current": -5010302000000.0,
    "Business income tax - deferred": 396075000000.0,
    "Business income tax expenses": -4614227000000.0,
    "Net profit/(loss) after tax": 18597344000000.0,
    "Minority interest": -15090000000.0,
    "Attributable to parent company": 18582254000000.0,
    "EPS basic (VND)": 4406.0,
    "EPS diluted (VND)": 0.0,
    "Interest and Similar Income": 67665496000000.0,
    "Interest and Similar Expenses": -33127768000000.0,
    "Net Interest Income": 34537728000000.0,
    "Net gain/(loss) from foreign currency and gold dealings": 3378274000000.0,
    "Net gain/(loss) from trading of trading securities": 145982000000.0,
    "Net gain/(loss) from disposal of investment securities": 7040000000.0,
    "Other Income": 3427795000000.0,
    "Other Expenses": -357970000000.0,
    "Net Other income/expenses": 3069825000000.0,
    "Dividends Income": 245096000000.0,
    "Total Operating Income": 45693391000000.0,
    "General and Admin Expenses": -15874542000000.0,
    "Net Operating Profit Before Allowance for Credit Loss": 29818849000000.0
  },
  {
    "report_period": "year",
    "ticker": "VCB",
    "Net Accounting Profit/(loss) before tax": 23049561000000.0,
    "Business income tax - current": -5081068000000.0,
    "Business income tax - deferred": 504025000000.0,
    "Business income tax expenses": -4577043000000.0,
    "Net profit/(loss) after tax": 18472518000000.0,
    "Minority interest": -21207000000.0,
    "Attributable to parent company": 18451311000000.0,
    "EPS basic (VND)": 4470.0,
    "EPS diluted (VND)": 0.0,
    "Interest and Similar Income": 69205134000000.0,
    "Interest and Similar Expenses": -32919659000000.0,
    "Net Interest Income": 36285475000000.0,
    "Net gain/(loss) from foreign currency and gold dealings": 3906399000000.0,
    "Net gain/(loss) from trading of trading securities": 1810000000.0,
    "Net gain/(loss) from disposal of investment securities": -98000000.0,
    "Other Income": 2544714000000.0,
    "Other Expenses": -744461000000.0,
    "Net Other income/expenses": 1800253000000.0,
    "Dividends Income": 461385000000.0,
    "Total Operating Income": 49062541000000.0,
    "General and Admin Expenses": -16038250000000.0,
    "Net Operating Profit Before Allowance for Credit Loss": 33024291000000.0
  }
]
```

#### Notes / caveats

Retrieve income statement data.

### note

- Kind: `method`
- Signature: `(period = None, lang = 'en', mode = 'final', style = 'readable', get_all = False, dropna = True, show_log = False) -> pd.DataFrame`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `vci`
- Return type: `pd.DataFrame`
- Purpose: Retrieve financial statement notes (thuyết minh báo cáo tài chính) if source is 'vci'.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `year` |
| `lang` | `POSITIONAL_OR_KEYWORD` | `False` | `en` | `` | `vi` |
| `mode` | `POSITIONAL_OR_KEYWORD` | `False` | `final` | `` | `omitted; default 'final'` |
| `style` | `POSITIONAL_OR_KEYWORD` | `False` | `readable` | `` | `omitted; default 'readable'` |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `True` |
| `dropna` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `` | `omitted; default True` |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `note` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `mas`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `note` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.financial.Finance`
- Provider method: `note`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:12.817939+00:00`
- Success: `True`
- Row count: `8`

```text
report_period, organCode, Mã CP, createDate, updateDate, yearReport, lengthReport, publicDate, Các khoản cho vay phân theo đối tượng khách hàng, Cho vay các tổ chức kinh tế, cá nhân trong nước, Chiết khấu thương phiếu và giấy tờ có giá, Cho thuê tài chính, Các khoản trả thay khách hàng, Cho vay bằng vốn tài trợ, uỷ thác đầu tư, Cho vay đối với các tổ chức, cá nhân nước ngoài, Cho vay theo chỉ định của Chính phủ, Nợ cho vay được khoanh và nợ chờ xử lý, Các khoản cho vay khác, Các khoản cho vay phân theo ngành, Thương mại, Nông nghiệp và lâm nghiệp, Sản xuất, Công nghiệp chế biến, chế tạo, Sản xuất và phân phối điện, khí đốt và nước nóng, hơi nước và điều hòa không khí, Cung cấp nước, quản lý và xử lý rác thải, nước thải, Khai khoáng, Xây dựng, Dịch vụ cộng đồng và cá nhân, Hoạt động phục vụ cá nhân và cộng đồng, Hoạt động các tổ chức và đoàn thể quốc tế, Y tế và hoạt động cứu trợ xã hội, Nghệ thuật, vui chơi, giải trí, Hoạt động hành chính và các dịch vụ hỗ trợ, Hoạt động của Đảng, tổ chức chính trị xã hội, quản lý nhà nước, an ninh quốc phòng, bảo đảm XH bắt buộc, Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất vật chất và dịch vụ tự tiêu dùng của hộ gia đình, Hoạt động dịch vụ khác, Kho bãi, vận tải, viễn thông, Vận tải, kho bãi, Thông tin và truyền thông, Giáo dục và đào tạo, Giáo dục và đào tạo, Giáo dục và đào tạo, Giáo dục và đào tạo, Hoạt động chuyên môn, khoa học và công nghệ, Bất động sản và tư vấn, Khách sạn và nhà hàng, Dịch vụ tài chính, Các ngành khác, Các khoản cho vay phân theo chất lượng nợ vay, Nợ đủ tiêu chuẩn, Nợ cần chú ý, Nợ dưới tiêu chuẩn, Nợ nghi ngờ, Nợ xấu có khả năng mất vốn, Các khoản cho vay phân theo thời gian, Cho vay ngắn hạn, Cho vay trung hạn, Cho vay dài hạn, Các khoản cho vay phân theo tiền tệ, VNĐ, VNĐ, Ngoại tệ và vàng, Các khoản cho vay phân theo vị trí địa lý, TP Hồ Chí Minh, Hà Nội, Đồng bằng sông Cửu Long, Miền trung, Khác, Khác, Khác, Các khoản cho vay phân theo nhóm khách hàng, Doanh nghiệp nhà nước, Doanh nghiệp nhà nước, Công ty TNHH và cổ phần, Doanh nghiệp nước ngoài, Doanh nghiệp nước ngoài, Hợp tác xã và công ty tư nhân, Cá nhân, Cá nhân, Khác, Khác, Khác, Các khoản tiền gửi phân theo loại tiền gửi, Tiền gửi không kỳ hạn, Tiền gửi có kỳ hạn, Tiền gửi tiết kiệm, Tiền gửi ký quỹ, Tiền gửi cho những mục đích riêng biệt, Các khoản tiền gửi phân theo loại tiền tệ, VNĐ, VNĐ, Ngoại tệ, Các khoản tiền gửi phân theo nhóm khách hàng, Doanh nghiệp nhà nước, Doanh nghiệp nhà nước, Doanh nghiệp tư nhân, Doanh nghiệp nước ngoài, Doanh nghiệp nước ngoài, Cá nhân, Cá nhân, Thu nhập lãi và các khoản Thu nhập tương tự, Tài sản sinh lãi, Công nợ phải trả lãi, Quỹ của tổ chức tín dụng, Quỹ dự trữ bổ sung vốn điều lệ, Quỹ dự phòng tài chính, Quỹ đầu tư phát triển, Khác, Khác, Khác, Thu nhập lãi và các khoản thu nhập tương tự, Thu nhập lãi cho vay khách hàng, Thu nhập lãi tiền gửi, Thu lãi từ kinh doanh, đầu tư chứng khoán nợ, Thu lãi từ chứng khoán kinh doanh, Thu lãi từ chứng khoán đầu tư, Thu nhập lãi cho thuê tài chính, Thu khác từ hoạt động tín dụng, Chi phí lãi và các chi phí tương tự, Trả lãi tiền gửi, Trả lãi tiền vay, Trả lãi phát hành trái phiếu và giấy tờ có giá, Chi phí khác cho hoạt động tín dụng, Lãi thuần từ hoạt động dịch vụ, Thu từ dịch vụ thanh toán, Thu từ dịch vụ ngân quỹ, Thu từ nghiệp vụ bảo lãnh, Thu từ nghiệp vụ ủy thác và đại lý, Thu từ hoạt động bảo hiểm, Thu từ dịch vụ môi giới, Thu khác, Chi về dịch vụ thanh toán, Chi về dịch vụ ngân quỹ, Chi về dịch vụ viễn thông, Chi về nghiệp vụ ủy thác và đại lý, Chi từ hoạt động bảo hiểm, Chi từ dịch vụ môi giới, Chi khác, Chi khác, Lãi thuần từ hoạt động kinh doanh ngoại hối, Thu từ kinh doanh ngoại tệ giao ngay, Thu từ các công cụ tài chính phái sinh tiền tệ, Thu từ giao dịch kinh doanh vàng, Lãi chênh lệch tỷ giá ngoại tệ kinh doanh, Chi về kinh doanh ngoại tệ giao ngay, Chi về các công cụ tài chính phái sinh tiền tệ, Chi về giao dịch kinh doanh vàng, Lỗ chênh lệch tỷ giá ngoại tệ kinh doanh, Lãi thuần từ mua bán chứng khoán kinh doanh, Thu nhập từ mua bán chứng khoán kinh doanh, Chi phí về mua bán chứng khoán kinh doanh, Hoàn nhập dự phòng giảm giá chứng khoán kinh doanh, Lãi thuần từ mua bán chứng khoán đầu tư, Thu nhập từ mua bán chứng khoán đầu tư, Lãi từ thanh lý các khoản đầu tư dài hạn khác, Chi phí về mua bán chứng khoán đầu tư, Lỗ do thanh lý các khoản đầu tư dài hạn khác, Hoàn nhập dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán, Hoàn nhập dự phòng giảm giá chứng khoán đầu tư giữ đến ngày đáo hạn, Chi phí hoạt động, Chi nộp thuế và các khoản phí, lệ phí, Chi phí cho nhân viên, Chi lương và phụ cấp, Các khoản chi đóng góp theo lương, Chi trợ cấp, Chi công tác xã hội, Chi công tác xã hội, Chi khác, Chi khác, Chi về tài sản, Chi Khấu hao TSCĐ, Chi khác về tài sản, Chi cho hoạt động quản lý công vụ, Chi nộp phí bảo hiểm, bảo toàn tiền gửi của khách hàng, Chi phí dự phòng giảm giá các khoản đầu tư dài hạn và dự phòng nợ khó đòi, Chi dự phòng trợ cấp thôi việc, Chi phí hoạt động khác, CAR, Chứng khoán kinh doanh, Chứng khoán nợ, Chứng khoán nợ, Chứng khoán nợ, Trái phiếu chính phủ, Trái phiếu chính phủ bảo lãnh, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Chứng khoán vốn, Chứng khoán vốn, Chứng khoán Vốn do các TCTD khác phát hành, Chứng khoán Vốn do các TCKT trong nước phát hành, Chứng khoán Vốn do các TCKT trong nước phát hành, Chứng khoán Vốn nước ngoài, Chứng khoán Vốn nước ngoài, Chứng khoán kinh doanh khác, Chứng khoán kinh doanh khác, Dự phòng rủi ro chứng khoán kinh doanh, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng chung, Dự phòng chung, Dự phòng chung, Dự phòng cụ thể, Dự phòng cụ thể, Dự phòng cụ thể, Tình trạng niêm yết của các chứng khoán kinh doanh, Chứng khoán nợ, Chứng khoán nợ, Chứng khoán nợ, Đã niêm yết, Đã niêm yết, Đã niêm yết, Chưa niêm yết, Chưa niêm yết, Chưa niêm yết, Chứng khoán vốn, Chứng khoán vốn, Đã niêm yết, Đã niêm yết, Đã niêm yết, Chưa niêm yết, Chưa niêm yết, Chưa niêm yết, Chứng khoán kinh doanh khác, Chứng khoán kinh doanh khác, Đã niêm yết, Đã niêm yết, Đã niêm yết, Chưa niêm yết, Chưa niêm yết, Chưa niêm yết, Chứng khoán đầu tư, Chứng khoán đầu tư sẵn sàng để bán, Chứng khoán Nợ, Trái phiếu Chính phủ, Trái phiếu Chính phủ, Trái phiếu do chính phủ bảo lãnh, Trái phiếu do chính phủ bảo lãnh, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Chứng khoán Vốn, Chứng khoán Vốn do các TCTD khác trong nước phát hành, Chứng khoán Vốn do các TCKT trong nước phát hành, Chứng khoán Vốn do các TCKT trong nước phát hành, Chứng khoán Vốn nước ngoài, Chứng khoán Vốn nước ngoài, Dự phòng rủi ro chứng khoán sẵn sàng để bán, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng chung, Dự phòng chung, Dự phòng chung, Dự phòng cụ thể, Dự phòng cụ thể, Dự phòng cụ thể, Chứng khoán đầu tư giữ đến ngày đáo hạn, Chứng khoán nợ, Chứng khoán nợ, Chứng khoán nợ, Trái phiếu Chính phủ, Trái phiếu Chính phủ, Trái phiếu do chính phủ bảo lãnh, Trái phiếu do chính phủ bảo lãnh, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCTD khác trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Trái phiếu do các TCKT trong nước phát hành, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Chứng khoán Nợ nước ngoài, Trái phiếu đặc biệt do VAMC phát hành, Dự phòng rủi ro chứng khoán đầu tư giữ đến ngày đáo hạn, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng giảm giá, Dự phòng chung, Dự phòng chung, Dự phòng chung, Dự phòng cụ thể, Dự phòng cụ thể, Dự phòng cụ thể, Dự phòng trái phiếu đặc biệt, Lãi thuần từ hoạt động khác, Thu nhập từ các khoản cho vay đã xử lý bằng quỹ dự phòng rủi ro, Thu từ thanh lý tài sản cố định, Thu từ nghiệp vụ mua bán nợ, Thu về nghiệp vụ tất toán Trái phiếu VAMC, Thu nhập về các công cụ tài chính phái sinh khác, Thu nhập khác, Chi phí cho nghiệp vụ hoán đổi lãi suất, Chi từ thanh lý tài sản, Chi về nghiệp vụ bán nợ, Chi về các công cụ tài chính phái sinh khác, Chi công tác xã hội, Chi công tác xã hội, Chi phí khác
```
- Dtypes: `{'report_period': 'str', 'organCode': 'str', 'Mã CP': 'str', 'createDate': 'str', 'updateDate': 'str', 'yearReport': 'int64', 'lengthReport': 'int64', 'publicDate': 'str', 'Các khoản cho vay phân theo đối tượng khách hàng': 'float64', 'Cho vay các tổ chức kinh tế, cá nhân trong nước': 'float64', 'Chiết khấu thương phiếu và giấy tờ có giá': 'float64', 'Cho thuê tài chính': 'float64', 'Các khoản trả thay khách hàng': 'float64', 'Cho vay bằng vốn tài trợ, uỷ thác đầu tư': 'float64', 'Cho vay đối với các tổ chức, cá nhân nước ngoài': 'float64', 'Cho vay theo chỉ định của Chính phủ': 'float64', 'Nợ cho vay được khoanh và nợ chờ xử lý': 'float64', 'Các khoản cho vay khác': 'float64', 'Các khoản cho vay phân theo ngành': 'float64', 'Thương mại': 'float64', 'Nông nghiệp và lâm nghiệp': 'float64', 'Sản xuất': 'float64', 'Công nghiệp chế biến, chế tạo': 'float64', 'Sản xuất và phân phối điện, khí đốt và nước nóng, hơi nước và điều hòa không khí': 'float64', 'Cung cấp nước, quản lý và xử lý rác thải, nước thải': 'float64', 'Khai khoáng': 'float64', 'Xây dựng': 'float64', 'Dịch vụ cộng đồng và cá nhân': 'float64', 'Hoạt động phục vụ cá nhân và cộng đồng': 'float64', 'Hoạt động các tổ chức và đoàn thể quốc tế': 'float64', 'Y tế và hoạt động cứu trợ xã hội': 'float64', 'Nghệ thuật, vui chơi, giải trí': 'float64', 'Hoạt động hành chính và các dịch vụ hỗ trợ': 'float64', 'Hoạt động của Đảng, tổ chức chính trị xã hội, quản lý nhà nước, an ninh quốc phòng, bảo đảm XH bắt buộc': 'float64', 'Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất vật chất và dịch vụ tự tiêu dùng của hộ gia đình': 'float64', 'Hoạt động dịch vụ khác': 'float64', 'Kho bãi, vận tải, viễn thông': 'float64', 'Vận tải, kho bãi': 'float64', 'Thông tin và truyền thông': 'float64', 'Giáo dục và đào tạo': 'float64', 'Hoạt động chuyên môn, khoa học và công nghệ': 'float64', 'Bất động sản và tư vấn': 'float64', 'Khách sạn và nhà hàng': 'float64', 'Dịch vụ tài chính': 'float64', 'Các ngành khác': 'float64', 'Các khoản cho vay phân theo chất lượng nợ vay': 'float64', 'Nợ đủ tiêu chuẩn': 'float64', 'Nợ cần chú ý': 'float64', 'Nợ dưới tiêu chuẩn': 'float64', 'Nợ nghi ngờ': 'float64', 'Nợ xấu có khả năng mất vốn': 'float64', 'Các khoản cho vay phân theo thời gian': 'float64', 'Cho vay ngắn hạn': 'float64', 'Cho vay trung hạn': 'float64', 'Cho vay dài hạn': 'float64', 'Các khoản cho vay phân theo tiền tệ': 'float64', 'VNĐ': 'float64', 'Ngoại tệ và vàng': 'float64', 'Các khoản cho vay phân theo vị trí địa lý': 'float64', 'TP Hồ Chí Minh': 'float64', 'Hà Nội': 'float64', 'Đồng bằng sông Cửu Long': 'float64', 'Miền trung': 'float64', 'Khác': 'float64', 'Các khoản cho vay phân theo nhóm khách hàng': 'float64', 'Doanh nghiệp nhà nước': 'float64', 'Công ty TNHH và cổ phần': 'float64', 'Doanh nghiệp nước ngoài': 'float64', 'Hợp tác xã và công ty tư nhân': 'float64', 'Cá nhân': 'float64', 'Các khoản tiền gửi phân theo loại tiền gửi': 'float64', 'Tiền gửi không kỳ hạn': 'float64', 'Tiền gửi có kỳ hạn': 'float64', 'Tiền gửi tiết kiệm': 'float64', 'Tiền gửi ký quỹ': 'float64', 'Tiền gửi cho những mục đích riêng biệt': 'float64', 'Các khoản tiền gửi phân theo loại tiền tệ': 'float64', 'Ngoại tệ': 'float64', 'Các khoản tiền gửi phân theo nhóm khách hàng': 'float64', 'Doanh nghiệp tư nhân': 'float64', 'Thu nhập lãi và các khoản Thu nhập tương tự': 'float64', 'Tài sản sinh lãi': 'float64', 'Công nợ phải trả lãi': 'float64', 'Quỹ của tổ chức tín dụng': 'float64', 'Quỹ dự trữ bổ sung vốn điều lệ': 'float64', 'Quỹ dự phòng tài chính': 'float64', 'Quỹ đầu tư phát triển': 'float64', 'Thu nhập lãi và các khoản thu nhập tương tự': 'float64', 'Thu nhập lãi cho vay khách hàng': 'float64', 'Thu nhập lãi tiền gửi': 'float64', 'Thu lãi từ kinh doanh, đầu tư chứng khoán nợ': 'float64', 'Thu lãi từ chứng khoán kinh doanh': 'float64', 'Thu lãi từ chứng khoán đầu tư': 'float64', 'Thu nhập lãi cho thuê tài chính': 'float64', 'Thu khác từ hoạt động tín dụng': 'float64', 'Chi phí lãi và các chi phí tương tự': 'float64', 'Trả lãi tiền gửi': 'float64', 'Trả lãi tiền vay': 'float64', 'Trả lãi phát hành trái phiếu và giấy tờ có giá': 'float64', 'Chi phí khác cho hoạt động tín dụng': 'float64', 'Lãi thuần từ hoạt động dịch vụ': 'float64', 'Thu từ dịch vụ thanh toán': 'float64', 'Thu từ dịch vụ ngân quỹ': 'float64', 'Thu từ nghiệp vụ bảo lãnh': 'float64', 'Thu từ nghiệp vụ ủy thác và đại lý': 'float64', 'Thu từ hoạt động bảo hiểm': 'float64', 'Thu từ dịch vụ môi giới': 'float64', 'Thu khác': 'float64', 'Chi về dịch vụ thanh toán': 'float64', 'Chi về dịch vụ ngân quỹ': 'float64', 'Chi về dịch vụ viễn thông': 'float64', 'Chi về nghiệp vụ ủy thác và đại lý': 'float64', 'Chi từ hoạt động bảo hiểm': 'float64', 'Chi từ dịch vụ môi giới': 'float64', 'Chi khác': 'float64', 'Lãi thuần từ hoạt động kinh doanh ngoại hối': 'float64', 'Thu từ kinh doanh ngoại tệ giao ngay': 'float64', 'Thu từ các công cụ tài chính phái sinh tiền tệ': 'float64', 'Thu từ giao dịch kinh doanh vàng': 'float64', 'Lãi chênh lệch tỷ giá ngoại tệ kinh doanh': 'float64', 'Chi về kinh doanh ngoại tệ giao ngay': 'float64', 'Chi về các công cụ tài chính phái sinh tiền tệ': 'float64', 'Chi về giao dịch kinh doanh vàng': 'float64', 'Lỗ chênh lệch tỷ giá ngoại tệ kinh doanh': 'float64', 'Lãi thuần từ mua bán chứng khoán kinh doanh': 'float64', 'Thu nhập từ mua bán chứng khoán kinh doanh': 'float64', 'Chi phí về mua bán chứng khoán kinh doanh': 'float64', 'Hoàn nhập dự phòng giảm giá chứng khoán kinh doanh': 'float64', 'Lãi thuần từ mua bán chứng khoán đầu tư': 'float64', 'Thu nhập từ mua bán chứng khoán đầu tư': 'float64', 'Lãi từ thanh lý các khoản đầu tư dài hạn khác': 'float64', 'Chi phí về mua bán chứng khoán đầu tư': 'float64', 'Lỗ do thanh lý các khoản đầu tư dài hạn khác': 'float64', 'Hoàn nhập dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán': 'float64', 'Hoàn nhập dự phòng giảm giá chứng khoán đầu tư giữ đến ngày đáo hạn': 'float64', 'Chi phí hoạt động': 'float64', 'Chi nộp thuế và các khoản phí, lệ phí': 'float64', 'Chi phí cho nhân viên': 'float64', 'Chi lương và phụ cấp': 'float64', 'Các khoản chi đóng góp theo lương': 'float64', 'Chi trợ cấp': 'float64', 'Chi công tác xã hội': 'float64', 'Chi về tài sản': 'float64', 'Chi Khấu hao TSCĐ': 'float64', 'Chi khác về tài sản': 'float64', 'Chi cho hoạt động quản lý công vụ': 'float64', 'Chi nộp phí bảo hiểm, bảo toàn tiền gửi của khách hàng': 'float64', 'Chi phí dự phòng giảm giá các khoản đầu tư dài hạn và dự phòng nợ khó đòi': 'float64', 'Chi dự phòng trợ cấp thôi việc': 'float64', 'Chi phí hoạt động khác': 'float64', 'CAR': 'float64', 'Chứng khoán kinh doanh': 'float64', 'Chứng khoán nợ': 'float64', 'Trái phiếu chính phủ': 'float64', 'Trái phiếu chính phủ bảo lãnh': 'float64', 'Trái phiếu do các TCTD khác trong nước phát hành': 'float64', 'Trái phiếu do các TCKT trong nước phát hành': 'float64', 'Chứng khoán Nợ nước ngoài': 'float64', 'Chứng khoán vốn': 'float64', 'Chứng khoán Vốn do các TCTD khác phát hành': 'float64', 'Chứng khoán Vốn do các TCKT trong nước phát hành': 'float64', 'Chứng khoán Vốn nước ngoài': 'float64', 'Chứng khoán kinh doanh khác': 'float64', 'Dự phòng rủi ro chứng khoán kinh doanh': 'float64', 'Dự phòng giảm giá': 'float64', 'Dự phòng chung': 'float64', 'Dự phòng cụ thể': 'float64', 'Tình trạng niêm yết của các chứng khoán kinh doanh': 'float64', 'Đã niêm yết': 'float64', 'Chưa niêm yết': 'float64', 'Chứng khoán đầu tư': 'float64', 'Chứng khoán đầu tư sẵn sàng để bán': 'float64', 'Chứng khoán Nợ': 'float64', 'Trái phiếu Chính phủ': 'float64', 'Trái phiếu do chính phủ bảo lãnh': 'float64', 'Chứng khoán Vốn': 'float64', 'Chứng khoán Vốn do các TCTD khác trong nước phát hành': 'float64', 'Dự phòng rủi ro chứng khoán sẵn sàng để bán': 'float64', 'Chứng khoán đầu tư giữ đến ngày đáo hạn': 'float64', 'Trái phiếu đặc biệt do VAMC phát hành': 'float64', 'Dự phòng rủi ro chứng khoán đầu tư giữ đến ngày đáo hạn': 'float64', 'Dự phòng trái phiếu đặc biệt': 'float64', 'Lãi thuần từ hoạt động khác': 'float64', 'Thu nhập từ các khoản cho vay đã xử lý bằng quỹ dự phòng rủi ro': 'float64', 'Thu từ thanh lý tài sản cố định': 'float64', 'Thu từ nghiệp vụ mua bán nợ': 'float64', 'Thu về nghiệp vụ tất toán Trái phiếu VAMC': 'float64', 'Thu nhập về các công cụ tài chính phái sinh khác': 'float64', 'Thu nhập khác': 'float64', 'Chi phí cho nghiệp vụ hoán đổi lãi suất': 'float64', 'Chi từ thanh lý tài sản': 'float64', 'Chi về nghiệp vụ bán nợ': 'float64', 'Chi về các công cụ tài chính phái sinh khác': 'float64', 'Chi phí khác': 'float64'}`

```json
[
  {
    "report_period": "year",
    "organCode": "VCB",
    "Mã CP": "VCB",
    "createDate": "2019-01-23T10:48:55.303",
    "updateDate": "2026-01-05T10:03:42.703",
    "yearReport": 2018,
    "lengthReport": 5,
    "publicDate": "2019-06-30T00:00:00",
    "Các khoản cho vay phân theo đối tượng khách hàng": 631866758000000.0,
    "Cho vay các tổ chức kinh tế, cá nhân trong nước": 624073743000000.0,
    "Chiết khấu thương phiếu và giấy tờ có giá": 3930917000000.0,
    "Cho thuê tài chính": 3855993000000.0,
    "Các khoản trả thay khách hàng": 1000000000.0,
    "Cho vay bằng vốn tài trợ, uỷ thác đầu tư": 0.0,
    "Cho vay đối với các tổ chức, cá nhân nước ngoài": 5105000000.0,
    "Cho vay theo chỉ định của Chính phủ": 0.0,
    "Nợ cho vay được khoanh và nợ chờ xử lý": 0.0,
    "Các khoản cho vay khác": 0.0,
    "Các khoản cho vay phân theo ngành": 631866758000000.0,
    "Thương mại": 120238625000000.0,
    "Nông nghiệp và lâm nghiệp": 14499324000000.0,
    "Sản xuất": 208551242000000.0,
    "Công nghiệp chế biến, chế tạo": 163734487000000.0,
    "Sản xuất và phân phối điện, khí đốt và nước nóng, hơi nước và điều hòa không khí": 29340404000000.0,
    "Cung cấp nước, quản lý và xử lý rác thải, nước thải": 0.0,
    "Khai khoáng": 15476351000000.0,
    "Xây dựng": 28873357000000.0,
    "Dịch vụ cộng đồng và cá nhân": 0.0,
    "Hoạt động phục vụ cá nhân và cộng đồng": 0.0,
    "Hoạt động các tổ chức và đoàn thể quốc tế": 0.0,
    "Y tế và hoạt động cứu trợ xã hội": 0.0,
    "Nghệ thuật, vui chơi, giải trí": 0.0,
    "Hoạt động hành chính và các dịch vụ hỗ trợ": 0.0,
    "Hoạt động của Đảng, tổ chức chính trị xã hội, quản lý nhà nước, an ninh quốc phòng, bảo đảm XH bắt buộc": 0.0,
    "Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất vật chất và dịch vụ tự tiêu dùng của hộ gia đình": 0.0,
    "Hoạt động dịch vụ khác": 0.0,
    "Kho bãi, vận tải, viễn thông": 23352261000000.0,
    "Vận tải, kho bãi": 23352261000000.0,
    "Thông tin và truyền thông": 0.0,
    "Giáo dục và đào tạo": 0.0,
    "Hoạt động chuyên môn, khoa học và công nghệ": 0.0,
    "Bất động sản và tư vấn": 0.0,
    "Khách sạn và nhà hàng": 11362643000000.0,
    "Dịch vụ tài chính": 0.0,
    "Các ngành khác": 224989306000000.0,
    "Các khoản cho vay phân theo chất lượng nợ vay": 631866758000000.0,
    "Nợ đủ tiêu chuẩn": 621862679000000.0,
    "Nợ cần chú ý": 3781086000000.0,
    "Nợ dưới tiêu chuẩn": 291788000000.0,
    "Nợ nghi ngờ": 1160507000000.0,
    "Nợ xấu có khả năng mất vốn": 4770698000000.0,
    "Các khoản cho vay phân theo thời gian": 631866758000000.0,
    "Cho vay ngắn hạn": 342212900000000.0,
    "Cho vay trung hạn": 53310111000000.0,
    "Cho vay dài hạn": 236343747000000.0,
    "Các khoản cho vay phân theo tiền tệ": 0.0,
    "VNĐ": 658636731000000.0,
    "Ngoại tệ và vàng": 0.0,
    "Các khoản cho vay phân theo vị trí địa lý": 0.0,
    "TP Hồ Chí Minh": 0.0,
    "Hà Nội": 0.0,
    "Đồng bằng sông Cửu Long": 0.0,
    "Miền trung": 0.0,
    "Khác": 0.0,
    "Các khoản cho vay phân theo nhóm khách hàng": 631866758000000.0,
    "Doanh nghiệp nhà nước": 0.0,
    "Công ty TNHH và cổ phần": 128333629000000.0,
    "Doanh nghiệp nước ngoài": 0.0,
    "Hợp tác xã và công ty tư nhân": 2487292000000.0,
    "Cá nhân": 421507009000000.0,
    "Các khoản tiền gửi phân theo loại tiền gửi": 801929115000000.0,
    "Tiền gửi không kỳ hạn": 226842211000000.0,
    "Tiền gửi có kỳ hạn": 558786377000000.0,
    "Tiền gửi tiết kiệm": 0.0,
    "Tiền gửi ký quỹ": 1351961000000.0,
    "Tiền gửi cho những mục đích riêng biệt": 14948566000000.0,
    "Các khoản tiền gửi phân theo loại tiền tệ": 801929115000000.0,
    "Ngoại tệ": 143292384000000.0,
    "Các khoản tiền gửi phân theo nhóm khách hàng": 801929115000000.0,
    "Doanh nghiệp tư nhân": 0.0,
    "Thu nhập lãi và các khoản Thu nhập tương tự": 380422106000000.0,
    "Tài sản sinh lãi": 0.0,
    "Công nợ phải trả lãi": 0.0,
    "Quỹ của tổ chức tín dụng": 9445732000000.0,
    "Quỹ dự trữ bổ sung vốn điều lệ": 3119785000000.0,
    "Quỹ dự phòng tài chính": 6255286000000.0,
    "Quỹ đầu tư phát triển": 70661000000.0,
    "Thu nhập lãi và các khoản thu nhập tương tự": 55863951000000.0,
    "Thu nhập lãi cho vay khách hàng": 43756805000000.0,
    "Thu nhập lãi tiền gửi": 2880373000000.0,
    "Thu lãi từ kinh doanh, đầu tư chứng khoán nợ": 8304634000000.0,
    "Thu lãi từ chứng khoán kinh doanh": 140013000000.0,
    "Thu lãi từ chứng khoán đầu tư": 8164621000000.0,
    "Thu nhập lãi cho thuê tài chính": 287078000000.0,
    "Thu khác từ hoạt động tín dụng": 635061000000.0,
    "Chi phí lãi và các chi phí tương tự": 27455435000000.0,
    "Trả lãi tiền gửi": 25365310000000.0,
    "Trả lãi tiền vay": 544079000000.0,
    "Trả lãi phát hành trái phiếu và giấy tờ có giá": 1516041000000.0,
    "Chi phí khác cho hoạt động tín dụng": 30005000000.0,
    "Lãi thuần từ hoạt động dịch vụ": 3402492000000.0,
    "Thu từ dịch vụ thanh toán": 4590636000000.0,
    "Thu từ dịch vụ ngân quỹ": 245694000000.0,
    "Thu từ nghiệp vụ bảo lãnh": 0.0,
    "Thu từ nghiệp vụ ủy thác và đại lý": 6321000000.0,
    "Thu từ hoạt động bảo hiểm": 0.0,
    "Thu từ dịch vụ môi giới": 0.0,
    "Thu khác": 2179504000000.0,
    "Chi về dịch vụ thanh toán": -2907563000000.0,
    "Chi về dịch vụ ngân quỹ": -69372000000.0,
    "Chi về dịch vụ viễn thông": -100790000000.0,
    "Chi về nghiệp vụ ủy thác và đại lý": -746000000.0,
    "Chi từ hoạt động bảo hiểm": 0.0,
    "Chi từ dịch vụ môi giới": 0.0,
    "Chi khác": 144611000000.0,
    "Lãi thuần từ hoạt động kinh doanh ngoại hối": 2266429000000.0,
    "Thu từ kinh doanh ngoại tệ giao ngay": 4449872000000.0,
    "Thu từ các công cụ tài chính phái sinh tiền tệ": 450526000000.0,
    "Thu từ giao dịch kinh doanh vàng": 0.0,
    "Lãi chênh lệch tỷ giá ngoại tệ kinh doanh": 325992000000.0,
    "Chi về kinh doanh ngoại tệ giao ngay": -865920000000.0,
    "Chi về các công cụ tài chính phái sinh tiền tệ": -1627596000000.0,
    "Chi về giao dịch kinh doanh vàng": 0.0,
    "Lỗ chênh lệch tỷ giá ngoại tệ kinh doanh": -466445000000.0,
    "Lãi thuần từ mua bán chứng khoán kinh doanh": 250462000000.0,
    "Thu nhập từ mua bán chứng khoán kinh doanh": 539389000000.0,
    "Chi phí về mua bán chứng khoán kinh doanh": -299893000000.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán kinh doanh": 10966000000.0,
    "Lãi thuần từ mua bán chứng khoán đầu tư": 0.0,
    "Thu nhập từ mua bán chứng khoán đầu tư": 0.0,
    "Lãi từ thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Chi phí về mua bán chứng khoán đầu tư": 0.0,
    "Lỗ do thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán": 0.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư giữ đến ngày đáo hạn": 0.0,
    "Chi phí hoạt động": 13611094000000.0,
    "Chi nộp thuế và các khoản phí, lệ phí": 253632000000.0,
    "Chi phí cho nhân viên": 7677596000000.0,
    "Chi lương và phụ cấp": 6920065000000.0,
    "Các khoản chi đóng góp theo lương": 608943000000.0,
    "Chi trợ cấp": 3977000000.0,
    "Chi công tác xã hội": -150034000000.0,
    "Chi về tài sản": 2340006000000.0,
    "Chi Khấu hao TSCĐ": 797551000000.0,
    "Chi khác về tài sản": 1542455000000.0,
    "Chi cho hoạt động quản lý công vụ": 2728089000000.0,
    "Chi nộp phí bảo hiểm, bảo toàn tiền gửi của khách hàng": 526591000000.0,
    "Chi phí dự phòng giảm giá các khoản đầu tư dài hạn và dự phòng nợ khó đòi": 41859000000.0,
    "Chi dự phòng trợ cấp thôi việc": 0.0,
    "Chi phí hoạt động khác": 43321000000.0,
    "CAR": 0.1214,
    "Chứng khoán kinh doanh": 2654806000000.0,
    "Chứng khoán nợ": 114251030000000.0,
    "Trái phiếu chính phủ": 1181914000000.0,
    "Trái phiếu chính phủ bảo lãnh": 0.0,
    "Trái phiếu do các TCTD khác trong nước phát hành": 22601979000000.0,
    "Trái phiếu do các TCKT trong nước phát hành": 7681750000000.0,
    "Chứng khoán Nợ nước ngoài": 0.0,
    "Chứng khoán vốn": 357436000000.0,
    "Chứng khoán Vốn do các TCTD khác phát hành": 10000000.0,
    "Chứng khoán Vốn do các TCKT trong nước phát hành": 8190000000.0,
    "Chứng khoán Vốn nước ngoài": 0.0,
    "Chứng khoán kinh doanh khác": 0.0,
    "Dự phòng rủi ro chứng khoán kinh doanh": -70245000000.0,
    "Dự phòng giảm giá": 0.0,
    "Dự phòng chung": -187734000000.0,
    "Dự phòng cụ thể": 0.0,
    "Tình trạng niêm yết của các chứng khoán kinh doanh": 2725051000000.0,
    "Đã niêm yết": 0.0,
    "Chưa niêm yết": 0.0,
    "Chứng khoán đầu tư": 149296430000000.0,
    "Chứng khoán đầu tư sẵn sàng để bán": 35233134000000.0,
    "Chứng khoán Nợ": 35313069000000.0,
    "Trái phiếu Chính phủ": 83967301000000.0,
    "Trái phiếu do chính phủ bảo lãnh": 0.0,
    "Chứng khoán Vốn": 8190000000.0,
    "Chứng khoán Vốn do các TCTD khác trong nước phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán sẵn sàng để bán": -88125000000.0,
    "Chứng khoán đầu tư giữ đến ngày đáo hạn": 114063296000000.0,
    "Trái phiếu đặc biệt do VAMC phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán đầu tư giữ đến ngày đáo hạn": -187734000000.0,
    "Dự phòng trái phiếu đặc biệt": 0.0,
    "Lãi thuần từ hoạt động khác": 3234365000000.0,
    "Thu nhập từ các khoản cho vay đã xử lý bằng quỹ dự phòng rủi ro": 3272247000000.0,
    "Thu từ thanh lý tài sản cố định": 0.0,
    "Thu từ nghiệp vụ mua bán nợ": 0.0,
    "Thu về nghiệp vụ tất toán Trái phiếu VAMC": 0.0,
    "Thu nhập về các công cụ tài chính phái sinh khác": 0.0,
    "Thu nhập khác": 243657000000.0,
    "Chi phí cho nghiệp vụ hoán đổi lãi suất": -9117000000.0,
    "Chi từ thanh lý tài sản": 0.0,
    "Chi về nghiệp vụ bán nợ": -159000000.0,
    "Chi về các công cụ tài chính phái sinh khác": 0.0,
    "Chi phí khác": -122229000000.0
  },
  {
    "report_period": "year",
    "organCode": "VCB",
    "Mã CP": "VCB",
    "createDate": "2020-01-30T11:30:00",
    "updateDate": "2026-01-05T10:00:18.25",
    "yearReport": 2019,
    "lengthReport": 5,
    "publicDate": "2020-03-17T00:00:00",
    "Các khoản cho vay phân theo đối tượng khách hàng": 734706891000000.0,
    "Cho vay các tổ chức kinh tế, cá nhân trong nước": 726968213000000.0,
    "Chiết khấu thương phiếu và giấy tờ có giá": 3172630000000.0,
    "Cho thuê tài chính": 4429029000000.0,
    "Các khoản trả thay khách hàng": 1000000000.0,
    "Cho vay bằng vốn tài trợ, uỷ thác đầu tư": 0.0,
    "Cho vay đối với các tổ chức, cá nhân nước ngoài": 136019000000.0,
    "Cho vay theo chỉ định của Chính phủ": 0.0,
    "Nợ cho vay được khoanh và nợ chờ xử lý": 0.0,
    "Các khoản cho vay khác": 0.0,
    "Các khoản cho vay phân theo ngành": 734706891000000.0,
    "Thương mại": 131856583000000.0,
    "Nông nghiệp và lâm nghiệp": 16122626000000.0,
    "Sản xuất": 218903033000000.0,
    "Công nghiệp chế biến, chế tạo": 174032670000000.0,
    "Sản xuất và phân phối điện, khí đốt và nước nóng, hơi nước và điều hòa không khí": 30411521000000.0,
    "Cung cấp nước, quản lý và xử lý rác thải, nước thải": 0.0,
    "Khai khoáng": 14458842000000.0,
    "Xây dựng": 32357572000000.0,
    "Dịch vụ cộng đồng và cá nhân": 0.0,
    "Hoạt động phục vụ cá nhân và cộng đồng": 0.0,
    "Hoạt động các tổ chức và đoàn thể quốc tế": 0.0,
    "Y tế và hoạt động cứu trợ xã hội": 0.0,
    "Nghệ thuật, vui chơi, giải trí": 0.0,
    "Hoạt động hành chính và các dịch vụ hỗ trợ": 0.0,
    "Hoạt động của Đảng, tổ chức chính trị xã hội, quản lý nhà nước, an ninh quốc phòng, bảo đảm XH bắt buộc": 0.0,
    "Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất vật chất và dịch vụ tự tiêu dùng của hộ gia đình": 0.0,
    "Hoạt động dịch vụ khác": 0.0,
    "Kho bãi, vận tải, viễn thông": 24742604000000.0,
    "Vận tải, kho bãi": 24742604000000.0,
    "Thông tin và truyền thông": 0.0,
    "Giáo dục và đào tạo": 0.0,
    "Hoạt động chuyên môn, khoa học và công nghệ": 0.0,
    "Bất động sản và tư vấn": 0.0,
    "Khách sạn và nhà hàng": 12837989000000.0,
    "Dịch vụ tài chính": 0.0,
    "Các ngành khác": 297886484000000.0,
    "Các khoản cho vay phân theo chất lượng nợ vay": 734706891000000.0,
    "Nợ đủ tiêu chuẩn": 726358767000000.0,
    "Nợ cần chú ý": 2978196000000.0,
    "Nợ dưới tiêu chuẩn": 686839000000.0,
    "Nợ nghi ngờ": 153248000000.0,
    "Nợ xấu có khả năng mất vốn": 4529841000000.0,
    "Các khoản cho vay phân theo thời gian": 734706891000000.0,
    "Cho vay ngắn hạn": 384355979000000.0,
    "Cho vay trung hạn": 48461992000000.0,
    "Cho vay dài hạn": 301888920000000.0,
    "Các khoản cho vay phân theo tiền tệ": 0.0,
    "VNĐ": 783384698000000.0,
    "Ngoại tệ và vàng": 0.0,
    "Các khoản cho vay phân theo vị trí địa lý": 0.0,
    "TP Hồ Chí Minh": 0.0,
    "Hà Nội": 0.0,
    "Đồng bằng sông Cửu Long": 0.0,
    "Miền trung": 0.0,
    "Khác": 50000000.0,
    "Các khoản cho vay phân theo nhóm khách hàng": 734706891000000.0,
    "Doanh nghiệp nhà nước": 0.0,
    "Công ty TNHH và cổ phần": 139575487000000.0,
    "Doanh nghiệp nước ngoài": 0.0,
    "Hợp tác xã và công ty tư nhân": 2268308000000.0,
    "Cá nhân": 466841936000000.0,
    "Các khoản tiền gửi phân theo loại tiền gửi": 928450869000000.0,
    "Tiền gửi không kỳ hạn": 262977124000000.0,
    "Tiền gửi có kỳ hạn": 642710681000000.0,
    "Tiền gửi tiết kiệm": 0.0,
    "Tiền gửi ký quỹ": 1743958000000.0,
    "Tiền gửi cho những mục đích riêng biệt": 21019106000000.0,
    "Các khoản tiền gửi phân theo loại tiền tệ": 928450869000000.0,
    "Ngoại tệ": 145066171000000.0,
    "Các khoản tiền gửi phân theo nhóm khách hàng": 928450869000000.0,
    "Doanh nghiệp tư nhân": 0.0,
    "Thu nhập lãi và các khoản Thu nhập tương tự": 461608933000000.0,
    "Tài sản sinh lãi": 0.0,
    "Công nợ phải trả lãi": 0.0,
    "Quỹ của tổ chức tín dụng": 12186141000000.0,
    "Quỹ dự trữ bổ sung vốn điều lệ": 4041013000000.0,
    "Quỹ dự phòng tài chính": 8074467000000.0,
    "Quỹ đầu tư phát triển": 70611000000.0,
    "Thu nhập lãi và các khoản thu nhập tương tự": 67665496000000.0,
    "Thu nhập lãi cho vay khách hàng": 53100063000000.0,
    "Thu nhập lãi tiền gửi": 4412907000000.0,
    "Thu lãi từ kinh doanh, đầu tư chứng khoán nợ": 8893830000000.0,
    "Thu lãi từ chứng khoán kinh doanh": 84531000000.0,
    "Thu lãi từ chứng khoán đầu tư": 8809299000000.0,
    "Thu nhập lãi cho thuê tài chính": 326787000000.0,
    "Thu khác từ hoạt động tín dụng": 931909000000.0,
    "Chi phí lãi và các chi phí tương tự": 33127768000000.0,
    "Trả lãi tiền gửi": 31205637000000.0,
    "Trả lãi tiền vay": 337223000000.0,
    "Trả lãi phát hành trái phiếu và giấy tờ có giá": 1500039000000.0,
    "Chi phí khác cho hoạt động tín dụng": 84869000000.0,
    "Lãi thuần từ hoạt động dịch vụ": 4309446000000.0,
    "Thu từ dịch vụ thanh toán": 6199194000000.0,
    "Thu từ dịch vụ ngân quỹ": 164507000000.0,
    "Thu từ nghiệp vụ bảo lãnh": 0.0,
    "Thu từ nghiệp vụ ủy thác và đại lý": 1278000000.0,
    "Thu từ hoạt động bảo hiểm": 0.0,
    "Thu từ dịch vụ môi giới": 0.0,
    "Thu khác": 2640130000000.0,
    "Chi về dịch vụ thanh toán": -3877007000000.0,
    "Chi về dịch vụ ngân quỹ": -96571000000.0,
    "Chi về dịch vụ viễn thông": -114680000000.0,
    "Chi về nghiệp vụ ủy thác và đại lý": -3485000000.0,
    "Chi từ hoạt động bảo hiểm": 0.0,
    "Chi từ dịch vụ môi giới": 0.0,
    "Chi khác": 201748000000.0,
    "Lãi thuần từ hoạt động kinh doanh ngoại hối": 3378274000000.0,
    "Thu từ kinh doanh ngoại tệ giao ngay": 6397100000000.0,
    "Thu từ các công cụ tài chính phái sinh tiền tệ": 821324000000.0,
    "Thu từ giao dịch kinh doanh vàng": 0.0,
    "Lãi chênh lệch tỷ giá ngoại tệ kinh doanh": 562523000000.0,
    "Chi về kinh doanh ngoại tệ giao ngay": -3627773000000.0,
    "Chi về các công cụ tài chính phái sinh tiền tệ": -701037000000.0,
    "Chi về giao dịch kinh doanh vàng": 0.0,
    "Lỗ chênh lệch tỷ giá ngoại tệ kinh doanh": -73863000000.0,
    "Lãi thuần từ mua bán chứng khoán kinh doanh": 145982000000.0,
    "Thu nhập từ mua bán chứng khoán kinh doanh": 207337000000.0,
    "Chi phí về mua bán chứng khoán kinh doanh": -43098000000.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán kinh doanh": -18257000000.0,
    "Lãi thuần từ mua bán chứng khoán đầu tư": 7040000000.0,
    "Thu nhập từ mua bán chứng khoán đầu tư": 7220000000.0,
    "Lãi từ thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Chi phí về mua bán chứng khoán đầu tư": -1000000.0,
    "Lỗ do thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán": -179000000.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư giữ đến ngày đáo hạn": 0.0,
    "Chi phí hoạt động": 15874542000000.0,
    "Chi nộp thuế và các khoản phí, lệ phí": 358529000000.0,
    "Chi phí cho nhân viên": 8668273000000.0,
    "Chi lương và phụ cấp": 7806861000000.0,
    "Các khoản chi đóng góp theo lương": 654098000000.0,
    "Chi trợ cấp": 5566000000.0,
    "Chi công tác xã hội": -163238000000.0,
    "Chi về tài sản": 2701784000000.0,
    "Chi Khấu hao TSCĐ": 829204000000.0,
    "Chi khác về tài sản": 1872580000000.0,
    "Chi cho hoạt động quản lý công vụ": 3515461000000.0,
    "Chi nộp phí bảo hiểm, bảo toàn tiền gửi của khách hàng": 578981000000.0,
    "Chi phí dự phòng giảm giá các khoản đầu tư dài hạn và dự phòng nợ khó đòi": 8002000000.0,
    "Chi dự phòng trợ cấp thôi việc": 0.0,
    "Chi phí hoạt động khác": 43512000000.0,
    "CAR": 0.0934,
    "Chứng khoán kinh doanh": 1801126000000.0,
    "Chứng khoán nợ": 132271302000000.0,
    "Trái phiếu chính phủ": 994592000000.0,
    "Trái phiếu chính phủ bảo lãnh": 0.0,
    "Trái phiếu do các TCTD khác trong nước phát hành": 42593538000000.0,
    "Trái phiếu do các TCKT trong nước phát hành": 6679769000000.0,
    "Chứng khoán Nợ nước ngoài": 0.0,
    "Chứng khoán vốn": 326070000000.0,
    "Chứng khoán Vốn do các TCTD khác phát hành": 4705000000.0,
    "Chứng khoán Vốn do các TCKT trong nước phát hành": 8190000000.0,
    "Chứng khoán Vốn nước ngoài": 0.0,
    "Chứng khoán kinh doanh khác": 0.0,
    "Dự phòng rủi ro chứng khoán kinh doanh": -88502000000.0,
    "Dự phòng giảm giá": 0.0,
    "Dự phòng chung": -316399000000.0,
    "Dự phòng cụ thể": 0.0,
    "Tình trạng niêm yết của các chứng khoán kinh doanh": 1991861000000.0,
    "Đã niêm yết": 0.0,
    "Chưa niêm yết": 0.0,
    "Chứng khoán đầu tư": 167529689000000.0,
    "Chứng khoán đầu tư sẵn sàng để bán": 35574786000000.0,
    "Chứng khoán Nợ": 35690900000000.0,
    "Trái phiếu Chính phủ": 82997995000000.0,
    "Trái phiếu do chính phủ bảo lãnh": 0.0,
    "Chứng khoán Vốn": 8190000000.0,
    "Chứng khoán Vốn do các TCTD khác trong nước phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán sẵn sàng để bán": -124304000000.0,
    "Chứng khoán đầu tư giữ đến ngày đáo hạn": 131954903000000.0,
    "Trái phiếu đặc biệt do VAMC phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán đầu tư giữ đến ngày đáo hạn": -316399000000.0,
    "Dự phòng trái phiếu đặc biệt": 0.0,
    "Lãi thuần từ hoạt động khác": 3069825000000.0,
    "Thu nhập từ các khoản cho vay đã xử lý bằng quỹ dự phòng rủi ro": 3179526000000.0,
    "Thu từ thanh lý tài sản cố định": 0.0,
    "Thu từ nghiệp vụ mua bán nợ": 0.0,
    "Thu về nghiệp vụ tất toán Trái phiếu VAMC": 0.0,
    "Thu nhập về các công cụ tài chính phái sinh khác": 0.0,
    "Thu nhập khác": 248269000000.0,
    "Chi phí cho nghiệp vụ hoán đổi lãi suất": 0.0,
    "Chi từ thanh lý tài sản": 0.0,
    "Chi về nghiệp vụ bán nợ": -32000000.0,
    "Chi về các công cụ tài chính phái sinh khác": 0.0,
    "Chi phí khác": -194700000000.0
  },
  {
    "report_period": "year",
    "organCode": "VCB",
    "Mã CP": "VCB",
    "createDate": "2021-01-21T11:21:37.933",
    "updateDate": "2025-05-06T09:00:01.543",
    "yearReport": 2020,
    "lengthReport": 5,
    "publicDate": "2021-04-01T00:00:00",
    "Các khoản cho vay phân theo đối tượng khách hàng": 839788261000000.0,
    "Cho vay các tổ chức kinh tế, cá nhân trong nước": 832010220000000.0,
    "Chiết khấu thương phiếu và giấy tờ có giá": 2549713000000.0,
    "Cho thuê tài chính": 4608056000000.0,
    "Các khoản trả thay khách hàng": 0.0,
    "Cho vay bằng vốn tài trợ, uỷ thác đầu tư": 0.0,
    "Cho vay đối với các tổ chức, cá nhân nước ngoài": 620272000000.0,
    "Cho vay theo chỉ định của Chính phủ": 0.0,
    "Nợ cho vay được khoanh và nợ chờ xử lý": 0.0,
    "Các khoản cho vay khác": 0.0,
    "Các khoản cho vay phân theo ngành": 839788261000000.0,
    "Thương mại": 202773035000000.0,
    "Nông nghiệp và lâm nghiệp": 17069197000000.0,
    "Sản xuất": 233446508000000.0,
    "Công nghiệp chế biến, chế tạo": 178521411000000.0,
    "Sản xuất và phân phối điện, khí đốt và nước nóng, hơi nước và điều hòa không khí": 40333441000000.0,
    "Cung cấp nước, quản lý và xử lý rác thải, nước thải": 0.0,
    "Khai khoáng": 14591656000000.0,
    "Xây dựng": 71273525000000.0,
    "Dịch vụ cộng đồng và cá nhân": 0.0,
    "Hoạt động phục vụ cá nhân và cộng đồng": 0.0,
    "Hoạt động các tổ chức và đoàn thể quốc tế": 0.0,
    "Y tế và hoạt động cứu trợ xã hội": 0.0,
    "Nghệ thuật, vui chơi, giải trí": 0.0,
    "Hoạt động hành chính và các dịch vụ hỗ trợ": 0.0,
    "Hoạt động của Đảng, tổ chức chính trị xã hội, quản lý nhà nước, an ninh quốc phòng, bảo đảm XH bắt buộc": 0.0,
    "Hoạt động làm thuê các công việc trong các hộ gia đình, sản xuất vật chất và dịch vụ tự tiêu dùng của hộ gia đình": 0.0,
    "Hoạt động dịch vụ khác": 0.0,
    "Kho bãi, vận tải, viễn thông": 26843614000000.0,
    "Vận tải, kho bãi": 26843614000000.0,
    "Thông tin và truyền thông": 0.0,
    "Giáo dục và đào tạo": 0.0,
    "Hoạt động chuyên môn, khoa học và công nghệ": 0.0,
    "Bất động sản và tư vấn": 0.0,
    "Khách sạn và nhà hàng": 10166471000000.0,
    "Dịch vụ tài chính": 0.0,
    "Các ngành khác": 278215911000000.0,
    "Các khoản cho vay phân theo chất lượng nợ vay": 839788261000000.0,
    "Nợ đủ tiêu chuẩn": 831765014000000.0,
    "Nợ cần chú ý": 2793678000000.0,
    "Nợ dưới tiêu chuẩn": 668690000000.0,
    "Nợ nghi ngờ": 223292000000.0,
    "Nợ xấu có khả năng mất vốn": 4337587000000.0,
    "Các khoản cho vay phân theo thời gian": 839788261000000.0,
    "Cho vay ngắn hạn": 434373690000000.0,
    "Cho vay trung hạn": 43091944000000.0,
    "Cho vay dài hạn": 362322627000000.0,
    "Các khoản cho vay phân theo tiền tệ": 0.0,
    "VNĐ": 884600244000000.0,
    "Ngoại tệ và vàng": 0.0,
    "Các khoản cho vay phân theo vị trí địa lý": 0.0,
    "TP Hồ Chí Minh": 0.0,
    "Hà Nội": 0.0,
    "Đồng bằng sông Cửu Long": 0.0,
    "Miền trung": 0.0,
    "Khác": 0.0,
    "Các khoản cho vay phân theo nhóm khách hàng": 839788261000000.0,
    "Doanh nghiệp nhà nước": 0.0,
    "Công ty TNHH và cổ phần": 155046852000000.0,
    "Doanh nghiệp nước ngoài": 0.0,
    "Hợp tác xã và công ty tư nhân": 2653308000000.0,
    "Cá nhân": 509788506000000.0,
    "Các khoản tiền gửi phân theo loại tiền gửi": 1032113567000000.0,
    "Tiền gửi không kỳ hạn": 307026182000000.0,
    "Tiền gửi có kỳ hạn": 693604644000000.0,
    "Tiền gửi tiết kiệm": 0.0,
    "Tiền gửi ký quỹ": 4156820000000.0,
    "Tiền gửi cho những mục đích riêng biệt": 27325921000000.0,
    "Các khoản tiền gửi phân theo loại tiền tệ": 1032113567000000.0,
    "Ngoại tệ": 147513323000000.0,
    "Các khoản tiền gửi phân theo nhóm khách hàng": 1032113567000000.0,
    "Doanh nghiệp tư nhân": 0.0,
    "Thu nhập lãi và các khoản Thu nhập tương tự": 522325061000000.0,
    "Tài sản sinh lãi": 0.0,
    "Công nợ phải trả lãi": 0.0,
    "Quỹ của tổ chức tín dụng": 14925803000000.0,
    "Quỹ dự trữ bổ sung vốn điều lệ": 4961163000000.0,
    "Quỹ dự phòng tài chính": 9893979000000.0,
    "Quỹ đầu tư phát triển": 70661000000.0,
    "Thu nhập lãi và các khoản thu nhập tương tự": 69205134000000.0,
    "Thu nhập lãi cho vay khách hàng": 56056454000000.0,
    "Thu nhập lãi tiền gửi": 2362665000000.0,
    "Thu lãi từ kinh doanh, đầu tư chứng khoán nợ": 9405447000000.0,
    "Thu lãi từ chứng khoán kinh doanh": 207589000000.0,
    "Thu lãi từ chứng khoán đầu tư": 9197858000000.0,
    "Thu nhập lãi cho thuê tài chính": 330131000000.0,
    "Thu khác từ hoạt động tín dụng": 1050437000000.0,
    "Chi phí lãi và các chi phí tương tự": 32919659000000.0,
    "Trả lãi tiền gửi": 31150286000000.0,
    "Trả lãi tiền vay": 197705000000.0,
    "Trả lãi phát hành trái phiếu và giấy tờ có giá": 1503840000000.0,
    "Chi phí khác cho hoạt động tín dụng": 67828000000.0,
    "Lãi thuần từ hoạt động dịch vụ": 6607317000000.0,
    "Thu từ dịch vụ thanh toán": 6017661000000.0,
    "Thu từ dịch vụ ngân quỹ": 74593000000.0,
    "Thu từ nghiệp vụ bảo lãnh": 0.0,
    "Thu từ nghiệp vụ ủy thác và đại lý": 9289000000.0,
    "Thu từ hoạt động bảo hiểm": 0.0,
    "Thu từ dịch vụ môi giới": 0.0,
    "Thu khác": 4486620000000.0,
    "Chi về dịch vụ thanh toán": -3192493000000.0,
    "Chi về dịch vụ ngân quỹ": -109932000000.0,
    "Chi về dịch vụ viễn thông": -125174000000.0,
    "Chi về nghiệp vụ ủy thác và đại lý": -6107000000.0,
    "Chi từ hoạt động bảo hiểm": 0.0,
    "Chi từ dịch vụ môi giới": 0.0,
    "Chi khác": 239068000000.0,
    "Lãi thuần từ hoạt động kinh doanh ngoại hối": 3906399000000.0,
    "Thu từ kinh doanh ngoại tệ giao ngay": 6841473000000.0,
    "Thu từ các công cụ tài chính phái sinh tiền tệ": 656940000000.0,
    "Thu từ giao dịch kinh doanh vàng": 0.0,
    "Lãi chênh lệch tỷ giá ngoại tệ kinh doanh": 474545000000.0,
    "Chi về kinh doanh ngoại tệ giao ngay": -2913034000000.0,
    "Chi về các công cụ tài chính phái sinh tiền tệ": -1114048000000.0,
    "Chi về giao dịch kinh doanh vàng": 0.0,
    "Lỗ chênh lệch tỷ giá ngoại tệ kinh doanh": -39477000000.0,
    "Lãi thuần từ mua bán chứng khoán kinh doanh": 1810000000.0,
    "Thu nhập từ mua bán chứng khoán kinh doanh": 113299000000.0,
    "Chi phí về mua bán chứng khoán kinh doanh": -162191000000.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán kinh doanh": 50702000000.0,
    "Lãi thuần từ mua bán chứng khoán đầu tư": -98000000.0,
    "Thu nhập từ mua bán chứng khoán đầu tư": 0.0,
    "Lãi từ thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Chi phí về mua bán chứng khoán đầu tư": 0.0,
    "Lỗ do thanh lý các khoản đầu tư dài hạn khác": 0.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán": -98000000.0,
    "Hoàn nhập dự phòng giảm giá chứng khoán đầu tư giữ đến ngày đáo hạn": 0.0,
    "Chi phí hoạt động": 16038250000000.0,
    "Chi nộp thuế và các khoản phí, lệ phí": 246322000000.0,
    "Chi phí cho nhân viên": 8603051000000.0,
    "Chi lương và phụ cấp": 7674586000000.0,
    "Các khoản chi đóng góp theo lương": 683529000000.0,
    "Chi trợ cấp": 5868000000.0,
    "Chi công tác xã hội": -301184000000.0,
    "Chi về tài sản": 2902528000000.0,
    "Chi Khấu hao TSCĐ": 1168499000000.0,
    "Chi khác về tài sản": 1734029000000.0,
    "Chi cho hoạt động quản lý công vụ": 3603531000000.0,
    "Chi nộp phí bảo hiểm, bảo toàn tiền gửi của khách hàng": 631788000000.0,
    "Chi phí dự phòng giảm giá các khoản đầu tư dài hạn và dự phòng nợ khó đòi": 0.0,
    "Chi dự phòng trợ cấp thôi việc": 0.0,
    "Chi phí hoạt động khác": 51030000000.0,
    "CAR": 0.0956,
    "Chứng khoán kinh doanh": 1954061000000.0,
    "Chứng khoán nợ": 115382544000000.0,
    "Trái phiếu chính phủ": 1126417000000.0,
    "Trái phiếu chính phủ bảo lãnh": 0.0,
    "Trái phiếu do các TCTD khác trong nước phát hành": 51041344000000.0,
    "Trái phiếu do các TCKT trong nước phát hành": 5339752000000.0,
    "Chứng khoán Nợ nước ngoài": 0.0,
    "Chứng khoán vốn": 326070000000.0,
    "Chứng khoán Vốn do các TCTD khác phát hành": 29437000000.0,
    "Chứng khoán Vốn do các TCKT trong nước phát hành": 8190000000.0,
    "Chứng khoán Vốn nước ngoài": 0.0,
    "Chứng khoán kinh doanh khác": 0.0,
    "Dự phòng rủi ro chứng khoán kinh doanh": -37800000000.0,
    "Dự phòng giảm giá": 0.0,
    "Dự phòng chung": -406523000000.0,
    "Dự phòng cụ thể": 0.0,
    "Tình trạng niêm yết của các chứng khoán kinh doanh": 1991861000000.0,
    "Đã niêm yết": 0.0,
    "Chưa niêm yết": 0.0,
    "Chứng khoán đầu tư": 156931097000000.0,
    "Chứng khoán đầu tư sẵn sàng để bán": 41955076000000.0,
    "Chứng khoán Nợ": 42140641000000.0,
    "Trái phiếu Chính phủ": 59001448000000.0,
    "Trái phiếu do chính phủ bảo lãnh": 0.0,
    "Chứng khoán Vốn": 8190000000.0,
    "Chứng khoán Vốn do các TCTD khác trong nước phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán sẵn sàng để bán": -193755000000.0,
    "Chứng khoán đầu tư giữ đến ngày đáo hạn": 114976021000000.0,
    "Trái phiếu đặc biệt do VAMC phát hành": 0.0,
    "Dự phòng rủi ro chứng khoán đầu tư giữ đến ngày đáo hạn": -406523000000.0,
    "Dự phòng trái phiếu đặc biệt": 0.0,
    "Lãi thuần từ hoạt động khác": 1800253000000.0,
    "Thu nhập từ các khoản cho vay đã xử lý bằng quỹ dự phòng rủi ro": 2421725000000.0,
    "Thu từ thanh lý tài sản cố định": 0.0,
    "Thu từ nghiệp vụ mua bán nợ": 0.0,
    "Thu về nghiệp vụ tất toán Trái phiếu VAMC": 0.0,
    "Thu nhập về các công cụ tài chính phái sinh khác": 0.0,
    "Thu nhập khác": 122989000000.0,
    "Chi phí cho nghiệp vụ hoán đổi lãi suất": -117968000000.0,
    "Chi từ thanh lý tài sản": 0.0,
    "Chi về nghiệp vụ bán nợ": -171000000.0,
    "Chi về các công cụ tài chính phái sinh khác": 0.0,
    "Chi phí khác": -325138000000.0
  }
]
```

#### Notes / caveats

Retrieve financial statement notes (thuyết minh báo cáo tài chính) if source is 'vci'.

### ratio

- Kind: `method`
- Signature: `(period = None, limit = 12, include_metadata = False, display_mode = "<FieldDisplayMode.STD: 'std'>", show_log = False) -> Examples`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `Examples`
- Purpose: Retrieve financial ratios.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `12` | `` |  | `5` |  | Số kỳ báo cáo tối đa cần lấy. Mặc định 4. |
| `include_metadata` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `omitted; default False` |  | Bao gồm thông tin audit và unit trong rows. Mặc định False. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en`, `financial`, `kbs` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) # Register provider from app.lib.vnstock_data_alt.core.registry import ProviderRegistry ProviderRegistry.register('financial', 'kbs', Finance) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  |  |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.financial.Finance`
- Provider method: `ratio`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:13.658042+00:00`
- Success: `True`
- Row count: `32`

```text
item, item_id, 2025, 2024, 2023, 2022, 2021
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64', '2021': 'float64'}`

```json
[
  {
    "item": "Beta",
    "item_id": "beta",
    "2025": 0.67,
    "2024": 0.65,
    "2023": 0.75,
    "2022": 0.84,
    "2021": 1.1
  },
  {
    "item": "Giá trị sổ sách của cổ phiếu (BVPS)",
    "item_id": "book_value_per_share_bvps",
    "2025": 27231.3,
    "2024": 35105.74,
    "2023": 29524.06,
    "2022": 28662.57,
    "2021": 29439.21
  },
  {
    "item": "Tăng trưởng vốn điều lệ",
    "item_id": "charter_capital",
    "2025": 49.5,
    "2024": 0.0,
    "2023": 18.1,
    "2022": 27.6,
    "2021": 0.0
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.financial.Finance`
- Provider method: `ratio`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:19:14.553398+00:00`
- Success: `True`
- Row count: `14`

```text
period, year_period, Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS), Giá trị sổ sách của cổ phiếu (BVPS), Chỉ số giá thị trường trên thu nhập (P/E), Chỉ số giá thị trường trên giá trị sổ sách (P/B), Tỷ suất cổ tức, Beta
```
- Dtypes: `{'period': 'int64', 'year_period': 'int64', 'Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)': 'str', 'Giá trị sổ sách của cổ phiếu (BVPS)': 'str', 'Chỉ số giá thị trường trên thu nhập (P/E)': 'str', 'Chỉ số giá thị trường trên giá trị sổ sách (P/B)': 'str', 'Tỷ suất cổ tức': 'str', 'Beta': 'str'}`

```json
[
  {
    "period": 2025,
    "year_period": 2025,
    "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)": "4542",
    "Giá trị sổ sách của cổ phiếu (BVPS)": "27231",
    "Chỉ số giá thị trường trên thu nhập (P/E)": "12.66",
    "Chỉ số giá thị trường trên giá trị sổ sách (P/B)": "2.11",
    "Tỷ suất cổ tức": "0.01",
    "Beta": "0.67"
  },
  {
    "period": 2024,
    "year_period": 2024,
    "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)": "6053",
    "Giá trị sổ sách của cổ phiếu (BVPS)": "35106",
    "Chỉ số giá thị trường trên thu nhập (P/E)": "15.07",
    "Chỉ số giá thị trường trên giá trị sổ sách (P/B)": "2.6",
    "Tỷ suất cổ tức": "0",
    "Beta": "0.65"
  },
  {
    "period": 2023,
    "year_period": 2023,
    "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)": "6507",
    "Giá trị sổ sách của cổ phiếu (BVPS)": "29524",
    "Chỉ số giá thị trường trên thu nhập (P/E)": "12.34",
    "Chỉ số giá thị trường trên giá trị sổ sách (P/B)": "2.72",
    "Tỷ suất cổ tức": "0",
    "Beta": "0.75"
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.financial.Finance`
- Provider method: `ratio`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve financial ratios.
Supports provider kwargs like flatten_columns, separator, drop_levels, etc.
