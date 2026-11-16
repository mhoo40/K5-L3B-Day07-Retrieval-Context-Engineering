# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
    + **Nguồn**: OCTOBER 2026 - Commodity Markets Outlook
    + **Quyền sử dụng**: Nguồn mở
    + **Số trang**: 59 dùng được
    + **Chunk**: ~360 chunks

- Vector store, embedding model, phiên bản và số chiều:
    + **Vector store**: PostgreSQL
    + **Embedding model**: sentence-transformers
    + **Phiên bản**: all-MiniLM-L6-v2
    + **Dimensions**: 384

- Agent model, cách chạy và phân chia dev/test:
    + **Agent model**: gpt-4o-mini
    + **Cách chạy**: 
    
        Truy vấn tìm kiếm mỗi câu hỏi (top_k = 4 chunks/lượt), cho phép tự động lọc theo metadata và tự động viết lại truy vấn khi thiếu ý hoặc ngữ cảnh bị cắt ngang giữa hai đoạn.

        Giới hạn budget = 5,500 ký tự (~6-7 chunks) để đảm bảo đủ không gian chứa bằng chứng từ nhiều lượt tìm kiếm mà không làm tràn prompt.

    + **Phân chia**: 18 `dev` (15 có answer, 3 không có answer), 8 (5 có answer, 3 không có answer).

- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
    + *Có Evident*: Đáp án có trong nội dung, được gắn nhãn `gold` gồm `source_id`, `page` và đoạn trích evidence nguyên văn. Agent trả lời kèm trích dẫn *[`chunk-id`]*.
    + *Thiếu Evident*: Đáp án không tồn tại trong báo cáo. Agent kiểm tra qua nhiều lượt truy vấn xác nhận dữ liệu vắng mặt và kích hoạt cơ chế từ chối `status = "abstain"`.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | tfidf, `top_k`=3, no filter, no rerank, `budget`=800 chars | Baseline khớp từ khóa thuần túy với ngân sách ngữ cảnh nhỏ | dev | 11/15 | 13/15 | 1/15 | ~2.6 ms/câu / $0.0000 |
| Phương pháp 1 | tfidf, `top_k`=4, no filter, no rerank, `budget`=3000 chars |Mở rộng `budget`=3000 để tránh cắt cụt đoạn văn ~750 ký tự | dev | 11/15 | 12/15 | 12/15 | ~53 ms/câu / $0.0000 |
| Phương pháp 2 | dense, `top_k`=4, no filter, no rerank, `budget`=3000 chars | Chuyển sang tìm kiếm ngữ nghĩa và mở rộng `budget`=3000 để tránh cắt cụt đoạn văn ~750 ký tự | dev | 11/15 | 12/15 | 12/15 | ~53 ms/câu / $0.0000 |
| Phương pháp 3 | hybrid_rrf, `top_k`=4, no filter, RRF rerank (`k`=60), `budget`=3000 chars | Kết hợp khớp thuật ngữ chính xác của TF-IDF và ngữ nghĩa của Dense | dev | 11/15 | 13/15 | 13/15 | ~52 ms/câu / $0.0000 |
| Cải thiện 1 | gpt-4o-mini + hybrid_rrf, `top_k`=4, metadata filter : `section`, `max_searches`=3, `budget`=3000 chars | Thêm vòng lặp Agent tự sinh truy vấn và lọc section, nhưng prompt từ chối quá cao | dev | 10/15 | 13/15 | 10/15 | 55s total / $0.004638 |
| Cải thiện 2 | gpt-4o-mini + hybrid_rrf, `top_k`=4, metadata filter : `section`, `max_searches`=3, `budget`=5000 chars | Năng `budget` và cải thiện prompt | dev | 10/15 | 13/15 | 15/15 | 70.45s total / $0.007511 |
| Cấu hình đã chọn | Cải thiện 2 | Kết quả tốt | test | 1/5 | 4/5 | 5/5 | 29.93 s / $0.004026 |
| Baseline | tfidf, `top_k`=3, no filter, no rerank, `budget`=800 chars | | test | 1/5 | 3/5 | 0/5 | 2.58 ms/câu / $0|

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
    + **Tool**: `search (query, filter)` gọi bộ truy xuất `HybridRRFRetriever (hybrid_rrf)` kết hợp Reciprocal Rank Fusion (k = 60) giữa `LexicalRetriever` (IF IDF) và `DensePgVectorRetriever`, trả về `top_k` = 4 chunks mỗi lượt kèm điểm số RRF và metadata `section`, `page`, `source_id`.
    + Tối đa 3 lượt gọi mỗi câu hỏi.
    + **Cách chọn context**: Chunks được khử trùng lặp theo `chunk_id`, giữ nguyên thứ tự ưu tiên xếp hạng và đóng gói vào cửa sổ ngữ cảnh với `budget` = 5,500 ký tự. Mỗi chunk được định dạng kèm header [`chunk_id` | `source_url` | `page N`].
    + **Điều kiện dừng**: Controller xuất `{"action": "answer", ...}` khi đủ, `{"action": "answer", "answer": "", "citations": []}` sau khi đã tìm kiếm ít nhất 2 lượt và xác nhận dữ liệu vắng mặt, số lần gọi lại chạm `max_searches` thì trả về `abstain`.
    + **Xử lý lỗi**: 
        Nội dung văn bản trích xuất được cô lập như dữ liệu thuần túy -> Chống Prompt Injection.

        mọi `chunk_id` được trích dẫn phải nằm trong danh sách `context_ids` để xác thực định dạng, nếu có lỗi kết nối từ provider, ghi nhận status = "error" và tách riêng khỏi mẫu tính điểm.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent v1 (`budget`=3000, 1-turn abstention) / dev (n=18) | 18/18 | 1 lượt/câu | *0: 5*, *1: 1*, *2: 9*, TB: 1.27/2 | 11/11 | 10/10 | 3/3 | 5/15 |
| Agent v2 (`budget`=5500, multi-turn prompt) / dev (n=18) | 18/18 | 1.33 lượt/câu | *0: 0*, *1: 0*, *2: 15*, TB: 2/2 | 21/21 | 15/15 | 3/3 | 0/15 |
| Agent v2 Locked (`budget`=5500, multi-turn) / test (n=8) | 8/8 | 1.75 lượt/câu | *0: 0*, *1: 0*, *2: 5*, TB: 2/ 2 | 8/8 | 5/5 | 3/3 | 0/5 |


| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án (Trang 17) | 1 lượt `search("energy prices contribution CPI inflation G7 countries")` -> trả về 4 chunks (`37424c63...`, `ece8d379...`, `4f3e156d...`, `8ff0ab27...`) | *"Energy prices directly contributed about 1 percentage point to inflation across the G7 in recent months `[ece8d37917f9a55c13d60d99]`."* | **`answer_grade = 2`**, `citation_support = [True]`. Khớp chính xác số liệu "about 1 percentage point" trong chunk `ece8d379...` (trang 17). |
| `q17` | Có đáp án, vắt 2 đoạn (Trang 39) | 1 lượt `search("Arabica coffee prices rise 10 percent 2026Q3")` -> trả về 4 chunks (`3e1dd8e0...`, `63d3ab83...`, `bff03a50...`, `dcf4d37e...`) | Nêu đủ cả lo ngại nguồn cung do El Niño và động đất lớn tại Colombia làm gián đoạn cơ sở chế biến cà phê `[3e1dd8e0736b7385e470cfaf] [bff03a50b2cee08640909b4e]`. | **`answer_grade = 2`**, `citation_support = [True, True]`. Tổng hợp trọn vẹn bằng chứng nằm tách ở 2 chunk liền kề trên trang 39. |
| `q18` | Có đáp án, đa nguyên nhân (Trang 53) | 1 lượt `search("cobalt lithium prices drop 2026Q3 reasons")` -> trả về 4 chunks (`80be1151...`, `5ee207d4...`, `4b0360d5...`, `21942b07...`) | Giải thích đủ: lithium giảm 13% do tái khởi động mỏ ở Úc, nối lại xuất khẩu từ Zimbabwe, doanh số EV Trung Quốc chững lại; cobalt giảm 25% điều chỉnh sau đợt tăng trước dù hạn ngạch DRC vẫn siết nguồn cung `[80be1151688ad6439bdc014b] [5ee207d455a10f874e23ee7d]`. | **`answer_grade = 2`**, `citation_support = [True, True]`. Đầy đủ cả số liệu lẫn bối cảnh chính sách thương mại hạn chế ở trang 53. |
| `q19` | Có đáp án (Trang 14) | 1 lượt `search("Middle East natural gas output below 2025 level 2027 condition")` -> trả về 4 chunks (`8bece761...`, `221b06df...`, `afd011c6...`, `1d634b4fe...`) | *"Middle East natural gas output is expected to remain below its 2025 level in 2027 primarily due to damage to LNG trains in Qatar, despite an overall regional recovery in natural gas production elsewhere `[8bece761409082af8e9abfa3]`."* | **`answer_grade = 2`**, `citation_support = [True]`. Trích xuất chính xác nguyên nhân hư hại các dây chuyền LNG tại Qatar (trang 14). |
| `q20` | Có đáp án, lọc section (Trang 60) | 1 lượt `search("crude oil top three net volatility transmitters...", filter={"section": "Special Focus..."})` -> trả về 4 chunks (`9e38f885...`, `bdffa2a6...`, `f36359da...`, `9c0a85a7...`) | Giải thích dầu thô chỉ nằm trong top 3 ở 10% số ngày giao dịch vì đóng góp truyền dẫn biến động tập trung vào một số ít giai đoạn biến động cực đoan, khác với bạc và ngô kết nối liên tục `[9e38f885f553082cb1d840d8][bdffa2a6b7cd37019ea88ce2]`. | **`answer_grade = 2`**, `citation_support = [True, True]`. Agent tự động dùng metadata filter đúng chuyên mục *Special Focus* và tổng hợp đủ 2 chunk trang 60. |
| `u04` | Thiếu evidence (Câu bẫy) | 3 lượt `search` (`World Bank forecast global lithium carbonate prices 2028` -> `...details`) -> chạm `search_limit` | `"Không đủ bằng chứng để trả lời."` (`citations = []`) | **`abstention_correct = True`**. Báo cáo chỉ nêu biến động giá lithium quý 3/2026 (-13%), không có dự báo giá lithium carbonate ($/tấn) năm 2028. |
| `u05` | Thiếu evidence (Câu bẫy) | 3 lượt `search` (`European natural gas storage subsidies household utility bills Germany October 2026`...) -> chạm `search_limit` | `"Không đủ bằng chứng để trả lời."` (`citations = []`) | **`abstention_correct = True`**. Báo cáo có đề cập tồn kho và giá khí đốt châu Âu tháng 10/2026 nhưng không có dữ liệu trợ cấp hóa đơn hộ gia đình ở Đức. |
| `u06` | Thiếu evidence (Câu bẫy) | 3 lượt `search` (`global palm oil production September 2026 month-on-month percentage change`) -> chạm `search_limit` | `"Không đủ bằng chứng để trả lời."` (`citations = []`) | **`abstention_correct = True`**. Báo cáo chỉ dự báo giá dầu cọ theo năm 2026–2027 (+14% và +10%), không có tỷ lệ % thay đổi sản lượng theo tháng vào tháng 9/2026. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:

    + **Dữ liệu chuẩn hóa**: `starter/data/converted/chunks.json`, `starter/data/converted/questions.json`, `starter/data/converted/cases.json`

    + **Kết quả truy xuất & Agent trên dev**: `starter/results/dev-tfidf/`, `starter/results/dev-methods/`, `starter/results/dev-agent`/, `starter/results/dev-agent-v2/`

- Kết quả chính thức trên test:

    + **Trace** & **metadata**: `starter/results/test-agent/traces.json`, `starter/results/test-agent/run_meta.json`

    + **Nhãn kiểm toán thủ công**: `starter/results/test-agent/reviews.json`

    + **Báo cáo tổng hợp chỉ số**: `starter/results/test-agent/summary.json`

- Ví dụ lỗi, đánh đổi và giới hạn:

    + **Lỗi 1**: Từ chối sớm trên câu hỏi hai vế (q04, trang 19): Câu hỏi yêu cầu cả tổng nguồn vốn World Bank công bố ($50-60 tỷ) lẫn phần vốn đã giải ngân/cam kết ($20 tỷ). Ở Agent v1, lượt tìm kiếm 1 chỉ lấy được đoạn đầu nên mô hình kết luận thiếu bằng chứng và từ chối ngay sau 1 lượt. Ở Agent v2, mô hình tìm bổ sung và trích dẫn đồng thời cả hai chunk [5be1cd7b728ec9aa6692281a] và [4ace1e54721466784402df8d].

    + **Lỗi 2**: Bằng chứng bị cắt ngang ranh giới đoạn văn (q07 trang 28, q09 trang 31): Chiến lược chia đoạn by_paragraph(max_size=850) tách một câu giải thích hoàn chỉnh sang hai chunk liền kề (ví dụ q09 có chủ ngữ "By contrast, U.S." ở cuối chunk e456e045... và nguyên nhân giảm 7% do khí đốt/năng lượng tái tạo ở đầu chunk 85790c00...). Mở rộng budget=5500 giúp cả hai chunk liền kề cùng lọt vào cửa sổ ngữ cảnh để Agent tổng hợp đầy đủ.

    + **Đánh đổi**: `tfidf` chạy mất 2.5-5.3 ms/câu với chi phí $0, nhưng giảm xuống 60% Hit@3 trên test và không tự tổng hợp hay từ chối câu bẫy. `Agent v2` tăng độ trễ lên ~3.74 s/câu và chi phí ~$0.0005/câu, đổi lại đạt 100% grounded_answer (2.00/2.00) và 0% ảo giác.

    + **Giới hạn**: Không đọc được các biểu đồ dạng ảnh.

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
