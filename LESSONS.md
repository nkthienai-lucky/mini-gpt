# Mini-GPT — Bài học tổng hợp (Stage 2: LLM Internals)

> Tổng hợp bài học kỹ thuật và quy trình từ project Mini-GPT + BPE tokenizer (GĐ2 roadmap AI Engineer, tuần 4–7). Mọi số liệu dưới đây đều là kết quả đo thật trên máy này (CPU / RTX 5060, WSL2), có lệnh tái tạo kèm theo.

---

## 1. Project xây được gì

| Thành phần | Nội dung | Bằng chứng |
|---|---|---|
| Mini-GPT baseline | Decoder-only transformer viết tay: attention + MLP, pre-norm, residual — không dùng `nn.Transformer` | `model.py`, 26 test |
| Train dài 5.000 steps | Val loss **4.239 → 1.611** (best 1.6075 @ step 4500), cosine decay hết lịch | `checkpoints/shakespeare.pt` |
| Benchmark KV cache | CPU + GPU, cached vs uncached, median 3 lần chạy | README "Generate and benchmark" |
| BPE tokenizer | `BasicTokenizer` + `RegexTokenizer` (GPT-4 pattern) viết tay, không thư viện tokenizer | `bpe.py`, 11 test |
| Thí nghiệm nén | Char-level vs BPE trên cùng corpus: **1.000 → 0.515 → 0.496** token/char | `scripts/compare_tokenizers.py` |

Config baseline: block 128 · 4 layers · 4 heads · n_embd 128 · AdamW lr 3e-4 · batch 32.

---

## 2. Bài học về Transformer & Training

### 2.1. Loss giảm nói lên gì — và không nói lên gì
- 20 steps: loss 4.24 → 3.25, sample ra `To the the the t the the...` — model mới học được phân phối tần suất ký tự.
- 5.000 steps: loss 1.61, sample ra lời thoại có tên nhân vật (`ROMEO:`, `HENRY BOLINGBROKE`), xuống dòng theo dòng thơ, từ cổ (`thank'd`, `thee`) — nhưng câu vẫn không liên kết.
- **Bài học:** loss là tín hiệu cần nhưng chưa đủ; chất lượng "trông giống dữ liệu" chỉ đánh giá được bằng mắt trên sample thật. Char-level 4-layer sẽ dừng ở "giống cục bộ, sai câu" — muốn mạch lạc cần model lớn hơn + token hóa tốt hơn + train lâu hơn.

### 2.2. Context window là ràng buộc cứng, không phải gợi ý
- Prompt 6 ký tự + 128 block_size ⇒ tối đa 122 token sinh ra. Vượt ⇒ `ValueError`.
- Bug thật đã gặp: chạy CUDA smoke với sample 64 token + prompt vượt block_size mặc định 64 cũ ⇒ phải nâng lên 128 kèm regression test.
- **Bài học:** mọi sinh text đều là `prompt + sinh ra ≤ block_size`. Đây là lỗi runtime phổ biến nhất khi dùng model nhỏ.

### 2.3. KV cache — kiếm được gì, mất gì
Kết quả đo (median, 3 repeats):

| Device | Uncached | Cached | Chênh lệch |
|---|---:|---:|---:|
| CPU | 570 tok/s | 575 tok/s | +1% |
| RTX 5060 | 262 tok/s | 299 tok/s | **+14%** |

- Cache tiết kiệm **tính lại K/V cho cả prefix mỗi token mới** —(prompt xử lý 1 lần, mỗi bước chỉ tính Q/K/V cho token mới).
- Trên model bé, CPU gần như không hưởng lợi: phần tiết kiệm nhỏ hơn noise đo. GPU hưởng lợi hơn vì tránh được nhiều kernel launch.
- **Phát hiện ngược đời:** GPU chậm hơn CPU (262 vs 570 tok/s) với model này — model quá nhỏ nên **thời gian launch kernel CUDA lấn át thời gian tính toán**. GPU chỉ thắng khi mỗi op đủ "đậm".
- Giá của cache: memory ∝ layers × heads × head_dim × số token đã cache.

### 2.4. Attention là O(n²) — giải thích bằng code
`QKᵀ` chấm từng query với từng key ⇒ ma trận score `n×n`: tính `O(n²·d)`, nhớ `O(n²)`. Gấp đôi context ≈ gấp tư phần này. Đây là lý do context dài đắt đỏ và vì sao có FlashAttention/attention thưa.

---

## 3. Bài học về Tokenizer / BPE

### 3.1. Byte-level BPE: không bao giờ gãy, nhưng không phải magic
- Encode về **byte** (256 token gốc) nên text bất kỳ — kể cả ký tự chưa gặp, emoji, tiếng Việt — đều roundtrip **chính xác tuyệt đối**. Không có ký tự "ngoài vocab".
- Nhưng **chất lượng nén phụ thuộc hoàn toàn vào dữ liệu train**: tokenizer train trên Shakespeare (thuần ASCII) encode tiếng Việt thành từng raw byte (`�`) — "à" tốn 2 token, không có merge nào ghép chúng cả.
- Đó là lý do tokenizer production (Qwen, GPT-4) train merge trên nhiều ngôn ngữ: để `à`, `ễ` thành 1 token.

### 3.2. Số liệu nén đo được (corpus 1.115.394 chars)

| Tokenizer | ids | token/char |
|---|---:|---:|
| Char-level (baseline mini-gpt) | 1.115.394 | 1.000 |
| Basic BPE (vocab 500) | 574.057 | 0.515 |
| Regex BPE (vocab 500) | 553.176 | **0.496** |

- Cùng vocab 500, **Regex pre-split tốt hơn Basic** — không tốn merge học những chuỗi vô nghĩa xuyên ranh giới từ.
- Lợi ích kép cho model: chuỗi vào ngắn hơn 2 lần ⇒ context chứa nhiều nội dung hơn 2 lần, attention (O(n²)) rẻ hơn ~4 lần cho cùng lượng văn bản.

### 3.3. GPT-4 split pattern — chi tiết khiến tôi sai 2 lần
Hành vi THẬT của pattern (tôi đã sai kỳ vọng khi viết plan, sửa bằng chạy thật):
- `" 123"` tách thành `" "` + `"123"` (prefix sở hữu `?+` chỉ ghép với **chữ**, không ghép số).
- Dấu câu gắn với **từ theo SAU**: `"a,b a,b"` ⇒ parts `['a', ',b', ' a', ',b']` — không phải `['a', ',', 'b']` như intuition đầu tiên.
- **Bài học:** regex pattern của tokenizer là hành vi quan trọng cần pin bằng test — nhưng kỳ vọng trong test phải được **xác minh bằng output thật trước**, không chép từ trí nhớ.

---

## 4. Bài học quy trình (đắt giá nhất)

### 4.1. Test không phân biệt được bug thì vô giá trị bằng chứng
Test boundary đầu tiên của tôi assert `(97,98) not in merges` — cặp đó **không thể tồn tại** ở bất kỳ implementation nào (a luôn theo sau là comma) ⇒ test pass kể cả khi code sai. Sửa: pin **chính xác** merges `{(44,98):256, (32,97):257}` rồi **mutation-test**: giả lập implementation sai (đếm pair trên stream nối liền) ⇒ nó học `(97,44)` ⇒ test fail. Chỉ khi đó test mới có giá trị bắt bug.

### 4.2. Kỳ vọng test phải đến từ thực nghiệm, không từ trí nhớ
Plan của tôi sai kỳ vọng 2 chỗ (pattern split + cấu trúc part) — cả hai đều là thứ tôi "tưởng nhớ" thay vì chạy thử.Reviewer bắt được. Quy tắc rút ra: **pin behavior của thư viện ngoài vào test chỉ sau khi chạy nó thật.**

### 4.3. Lệnh trong README phải chạy được nguyên văn
`pytest -q` trong README fail (`No module named 'bpe'`) vì bare pytest không đưa repo root vào `sys.path` — chỉ `python -m pytest` mới chạy được. Lỗi có sẵn cả repo nhưng không ai phát hiện vì mọi người chạy bằng cách sau. Fix: `conftest.py` ở root chèn path. **Bài học:** README ghi lệnh nào phải verify lệnh đó nguyên văn, không phải "tương đương".

### 4.4. Review độc lập đáng giá ngay cả khi reviewer "hỏng"
Reviewer agent bị treo giữa chừng — nhưng 2 câu hỏi probe nó kịp đặt ra đều phát hiện issue thật (mục 4.1, 4.3). Câu hỏi đúng từ một người (hoặc agent) nhìn bằng mắt tươi tìm ra thứ tác giả bỏ qua.

### 4.5. Ghi nhận bằng số liệu, không bằng cảm tính
Mọi claim trong README đều có bảng số đi kèm: loss theo step, tok/s cached/uncached trên 2 device, token/char của 3 tokenizer. "Nhanh hơn", "tốt hơn" không có chỗ đứng — "262 → 299 tok/s (+14%)" mới là tài liệu phỏng vấn có thể vertify.

---

## 5. Q&A phỏng vấn — trả lời bằng số liệu của chính project

**Q: Vì sao attention là O(n²)?**
Mỗi query chấm mỗi key ⇒ ma trận score n×n; tính O(n²·d), nhớ O(n²). Trong code của tôi: `QKᵀ/√D` tạo tensor `(B, H, T, T)` — thấy trực tiếp kích thước n².

**Q: KV cache tiết kiệm gì?**
Không phải tính lại K/V của cả prefix mỗi token. Đo được: +14% tok/s trên RTX 5060 (262→299), ~0% trên CPU vì model nhỏ — cache có giá khi phần việc bỏ được lớn hơn overhead. Đổi bằng memory ∝ layers × heads × head_dim × tokens.

**Q: Tại sao chọn BPE thay vì char-level?**
Đo trên cùng corpus: char 1.000 token/char, BPE 0.496 — cùng nội dung dùng ít hơn 2 lần token ⇒ context hiệu dụng gấp đôi, attention rẻ gấp 4. Và byte-level không bao giờ gãy với text lạ.

**Q: Tokenizer của bạn xử lý tiếng Việt ra sao?**
Roundtrip chính xác 100% (byte-level), nhưng nén kém nếu train thiếu tiếng Việt — diacritics vỡ thành raw bytes. Giải pháp production: train merge đa ngôn ngữ.

**Q: Train đến đâu thì model "biết nói"?**
Loss 3.25 (20 steps): lặp ký tự phổ biến. Loss 1.61 (5.000 steps): cấu trúc lời thoại, tên nhân vật, ngắt dòng thơ — vẫn sai ngữ pháp câu. Scaling cần ở cả tokenizer, kích thước model và dữ liệu.

---

## 6. Backlog còn mở

- [ ] Train BPE trên dữ liệu tiếng Việt — khắc phục failure mode đã thấy
- [ ] RoPE so với learned positional embedding
- [ ] LoRA/QLoRA Qwen2.5-0.5B — chạy **trên RTX 5060 local** (QLoRA 4-bit cần ~5–6GB VRAM), không cần Colab như roadmap ghi (roadmap viết cho người không có GPU)
- [ ] Tích hợp `chatbot-cli` sau khi xong phần internals

---

*Tài liệu liên quan: [README.md](README.md) (kiến trúc + lệnh chạy) · [bpe.py](bpe.py) · [scripts/compare_tokenizers.py](scripts/compare_tokenizers.py) · Roadmap: `AI-Engineer-Roadmap/docs/cores/02-LLM-INTERNALS/`*
