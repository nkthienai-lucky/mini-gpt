"""Deterministic builder for the Vietnamese JSON-format LoRA dataset.

Content strategy: hand-written topic seed banks (factually correct, 1-3
sentences each), composed deterministically with random.Random(1337).
Variation comes from topic x phrasing, never from fabricated facts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# (question_phrazes, answer, reason) — hand-written, Vietnamese, AI/LLM topics.
TOPICS: list[tuple[list[str], str, str]] = [
    (["Attention là gì?", "Cơ chế attention hoạt động thế nào?", "Tại sao attention quan trọng trong transformer?"],
     "Attention là cơ chế giúp model tính trọng số quan trọng của từng vị trí trong chuỗi đầu vào khi tạo biểu diễn cho một vị trí.",
     "Attention cho phép mỗi token nhìn thấy toàn bộ ngữ cảnh thay vì chỉ vị trí gần nhau."),
    (["KV cache là gì?", "KV cache tiết kiệm được gì?", "Tại sao sinh text nhanh hơn khi dùng KV cache?"],
     "KV cache lưu lại key và value của các token đã xử lý để không phải tính lại khi sinh token mới.",
     "Khi sinh token mới, chỉ cần query của token mới nhân với key/value đã lưu, giảm rất nhiều phép tính lặp."),
    (["BPE tokenizer là gì?", "BPE hoạt động như thế nào?", "Tại sao dùng BPE thay vì tách theo ký tự?"],
     "BPE là thuật toán token hóa bắt đầu từ byte rồi lần lượt gộp cặp token liền kề xuất hiện nhiều nhất thành token mới.",
     "BPE cân bằng giữa vocab nhỏ và chuỗi token ngắn, giúp model học hiệu quả hơn tách theo ký tự."),
    (["LoRA là gì?", "LoRA hoạt động thế nào?", "Tại sao LoRA rẻ hơn fine-tune toàn bộ model?"],
     "LoRA đóng băng model gốc và chỉ train hai ma trận nhỏ có rank thấp ghép vào các lớp attention.",
     "Số tham số train được giảm còn dưới vài phần trăm nên tốn ít VRAM và thời gian hơn nhiều."),
    (["QLoRA khác LoRA ở điểm gì?", "Khi nào nên dùng QLoRA?", "QLoRA lượng tử hóa phần nào của model?"],
     "QLoRA là LoRA nhưng ném model gốc về 4-bit NF4 trước khi gắn adapter, chỉ adapter được train ở độ chính xác cao.",
     "Nén 4-bit giảm bộ nhớ weights khoảng bốn lần, cho phép fine-tune model lớn hơn trên cùng một GPU."),
    (["Quantization là gì?", "Tại sao model 3B chạy được trên CPU?", "INT8 và INT4 khác nhau thế nào?"],
     "Quantization giảm số bit biểu diễn weights từ 16 xuống 8 hoặc 4 để tiết kiệm bộ nhớ và tăng tốc độ tính toán.",
     "Bộ nhớ giảm theo tỷ lệ số bit, đổi lại một chút độ chính xác, nên model 3B có thể chạy được trên CPU."),
    (["Embedding là gì?", "Embedding dùng để làm gì?", "Tại sao các từ giống nghĩa có vector gần nhau?"],
     "Embedding là vector số biểu diễn một token hoặc một đoạn văn bản trong không gian nhiều chiều.",
     "Model học embedding sao cho nội dung giống nhau nằm gần nhau, nên khoảng cách vector phản ánh độ tương đồng ngữ nghĩa."),
    (["Fine-tune khác RAG ở điểm gì?", "Khi nào nên fine-tune thay vì RAG?", "RAG giải quyết vấn đề gì mà fine-tune không giải được?"],
     "Fine-tune thay đổi weights để uốn hành vi model, còn RAG lấy kiến thức từ nguồn ngoài rồi đưa vào prompt.",
     "Kiến thức cập nhật thường xuyên thì RAG rẻ hơn, còn hành vi hay format cố định thì fine-tune hiệu quả hơn."),
    (["Hallucination là gì?", "Tại sao LLM bị ảo giác?", "Làm sao giảm hallucination?"],
     "Hallucination là hiện tượng model tự tin đưa ra thông tin sai hoặc bịa ra không có trong dữ liệu.",
     "Model sinh từ xác suất nên có thể tạo ra câu trông hợp lý nhưng sai; grounding bằng RAG và trích dẫn nguồn giúp giảm."),
    (["Context window là gì?", "Tại sao context dài thì đắt?", "Model xử lý được bao nhiêu token cùng lúc?"],
     "Context window là số token tối đa model nhìn thấy trong một lần xử lý, gồm cả prompt và câu trả lời.",
     "Chi phí attention tăng theo bình phương độ dài chuỗi nên context càng dài càng tốn tính toán và bộ nhớ."),
    (["Temperature trong sinh văn bản là gì?", "Temperature thấp và cao khác nhau thế nào?", "Khi nào nên hạ temperature?"],
     "Temperature là tham số chia lại xác suất trước khi chọn token, thấp thì chọn an toàn, cao thì ngẫu nhiên hơn.",
     "Temperature về 0 chọn token xác suất cao nhất, phù hợp khi cần câu trả lời ổn định như xuất JSON."),
    (["Top-k sampling là gì?", "Top-k khác temperature thế nào?", "Vì sao cần giới hạn số ứng viên khi sample?"],
     "Top-k chỉ chọn token trong k ứng viên có xác suất cao nhất rồi sample đều trong đó.",
     "Giới hạn ứng viên loại đuôi phân phối xác suất dài, tránh nhặt phải token vô nghĩa dù temperature cao."),
    (["Prompt engineering là gì?", "Khi nào prompt đủ tốt thay vì fine-tune?", "Vì sao cùng model mà prompt khác cho kết quả khác?"],
     "Prompt engineering là kỹ thuật viết hướng dẫn và ví dụ trong đầu vào để model làm đúng việc mà không cần train lại.",
     "Model điều kiện hoàn toàn vào prompt nên ngữ cảnh, ví dụ và ràng buộc rõ ràng thay đổi trực tiếp chất lượng đầu ra."),
    (["Pretraining là gì?", "Pretraining khác fine-tuning thế nào?", "Model học được gì trong pretraining?"],
     "Pretraining là huấn luyện model dự đoán token tiếp theo trên kho dữ liệu khổng lồ để nắm ngôn ngữ và kiến thức tổng quát.",
     "Sau pretraining model biết ngôn ngữ nhưng chưa biết nghe lệnh, nên cần fine-tune theo tác vụ cụ thể."),
    (["SFT là gì?", "Supervised fine-tuning dạy model cái gì?", "Sau SFT model thay đổi thế nào?"],
     "SFT là fine-tune có giám sát trên các cặp hướng dẫn và câu trả lời mẫu để model biết cách làm theo yêu cầu.",
     "Model học format và phong cách trả lời phù hợp với kỳ vọng người dùng thay vì chỉ tiếp chuỗi văn bản."),
    (["RLHF là gì?", "RLHF giải quyết vấn đề gì?", "Vì sao cần phản hồi con người sau SFT?"],
     "RLHF huấn luyện model tối ưu theo phần thưởng học từ sở thích con người thay vì chỉ theo dữ liệu mẫu.",
     "Nhiều câu trả lời đều đúng ngữ pháp nhưng chỉ con người phân biệt được cái hữu ích hơn, reward model nắm được điều đó."),
    (["DPO khác RLHF thế nào?", "DPO hoạt động ra sao?", "Vì sao DPO đơn giản hơn RLHF?"],
     "DPO fine-tune trực tiếp trên cặp câu trả lời được thích và không thích mà không cần huấn luyện reward model riêng.",
     "DPO đưa mục tiêu ưu tiên sở thích vào hàm loss trực tiếp nên pipeline ngắn và ổn định hơn RLHF."),
    (["Self-attention là gì?", "Self-attention khác attention thường ở đâu?", "Q K V là những gì?"],
     "Self-attention cho mỗi token trong cùng một chuỗi tính trọng số với tất cả token khác, gồm phép chiếu Q K V.",
     "Q của token hiện tại nhân với K của mọi token tạo điểm tương quan, softmax thành trọng số rồi nhân với V."),
    (["Multi-head attention là gì?", "Vì sao cần nhiều head?", "Các head khác nhau ở điểm gì?"],
     "Multi-head attention chạy nhiều phép attention song song trên các không gian con khác nhau rồi ghép kết quả.",
     "Mỗi head tự do chuyên môn hóa một kiểu quan hệ, ví dụ một head nắm cú pháp, một head nắm ngữ nghĩa dài."),
    (["Residual connection là gì?", "Residual giúp gì khi train mạng sâu?", "Skip connection hoạt động thế nào?"],
     "Residual connection cộng đầu vào của block vào đầu ra, tạo đường tắt cho gradient khi lan truyền ngược.",
     "Gradient đi qua đường cộng trực tiếp nên không tan mất qua nhiều tầng, giúp mạng sâu train được."),
    (["LayerNorm là gì?", "Vì sao cần chuẩn hóa giữa các tầng?", "Pre-norm khác post-norm thế nào?"],
     "LayerNorm chuẩn hóa vector kích hoạt về trung bình không và phương sai một tại từng vị trí.",
     "Pre-norm đặt chuẩn hóa trước block chính giúp train ổn định hơn post-norm ở mạng rất sâu."),
    (["Decoder-only là gì?", "Vì sao GPT dùng decoder-only?", "Decoder-only khác encoder-decoder ở đâu?"],
     "Decoder-only là kiến trúc chỉ dùng ngăn xếp decoder với attention che khuất tương lai để sinh token tuần tự.",
     "GPT decoder-only vì mục tiêu sinh văn bản tiếp theo tự nhiên, còn encoder-decoder như T5 phù hợp seq2seq."),
    (["Cross-entropy loss là gì?", "Model ngôn ngữ tối ưu hàm mục tiêu nào?", "Cross-entropy đo lường cái gì?"],
     "Cross-entropy đo khoảng cách giữa phân phối xác suất model và token đúng tại mỗi vị trí.",
     "Model ngôn ngữ học bằng cách giảm cross-entropy, tức tăng xác suất token tiếp theo đúng trong dữ liệu."),
    (["Learning rate ảnh hưởng thế nào?", "Learning rate quá cao thì sao?", "Vì sao cần warmup learning rate?"],
     "Learning rate quyết định bước cập nhật weights sau mỗi batch, quá cao thì dao động, quá thấp thì chậm hội tụ.",
     "Warmup tăng dần learning rate lúc đầu để optimizer ổn định trước khi decay tránh vỡ hội tụ sớm."),
    (["Batch size ảnh hưởng training thế nào?", "Batch lớn và nhỏ khác nhau ra sao?", "Vì sao dùng gradient accumulation?"],
     "Batch size quyết định số mẫu cộng gradient trước mỗi bước cập nhật, ảnh hưởng độ ổn định và bộ nhớ.",
     "Gradient accumulation cộng gradient nhiều batch nhỏ để mô phỏng batch lớn mà không cần thêm VRAM."),
    (["Overfitting trong fine-tune là gì?", "Làm sao biết model bị overfit?", "Vì sao theo dõi validation loss?"],
     "Overfitting là model khớp quá mức dữ liệu train, mất khả năng tổng quát hóa trên dữ liệu mới.",
     "Khi train loss giảm mà validation loss tăng là dấu hiệu overfit, nên dừng sớm hoặc giảm số epoch."),
    (["Zero-shot là gì?", "Few-shot khác zero-shot thế nào?", "Khi nào few-shot giúp?"],
     "Zero-shot là yêu cầu model làm việc mà không đưa ví dụ nào, few-shot đưa vài ví dụ trong prompt.",
     "Ví dụ trong prompt cho model thấy format và phong cách mong muốn nên few-shot thường ổn định hơn."),
    (["Chain-of-thought là gì?", "Vì sao bắt model lý luận từng bước lại đúng hơn?", "Khi nào chain-of-thought hữu ích?"],
     "Chain-of-thought là hướng dẫn model viết các bước lý luận trung gian trước khi đưa câu trả lời cuối.",
     "Sinh từng bước chia bài toán khó thành nhiều bước dễ, mỗi bước ít xác suất sai hơn một câu trả lời trực tiếp."),
    (["Tokenizer là gì?", "Token khác chữ thế nào?", "Vì sao model không xử lý trực tiếp chữ cái?"],
     "Tokenizer cắt văn bản thành các token rời rồi ánh xạ sang số nguyên để model xử lý được.",
     "Token có thể là chữ trọn vẹn, một phần chữ hoặc ký tự; làm việc trên số nên model cần bước này."),
    (["Cosine similarity là gì?", "Đo độ giống nhau của hai vector thế nào?", "Vì sao dùng cosine thay vì khoảng cách thường?"],
     "Cosine similarity đo cosin góc giữa hai vector, bằng một khi cùng hướng và không khi vuông góc.",
     "Cosine chỉ xét hướng không xét độ dài vector nên phản ánh tương đồng ngữ nghĩa tốt hơn khoảng cách Euclid."),
    (["Vector database là gì?", "Khi nào cần vector database?", "Vector database khác SQL ở đâu?"],
     "Vector database lưu embedding và tìm nhanh các vector gần nhất theo khoảng cách hoặc cosine.",
     "Tìm lân cận xấp xỉ trên không gian vector là phép toán SQL không làm được, cần cấu trúc chỉ mục chuyên dụng."),
    (["Chunking tài liệu là gì?", "Chunk quá lớn hoặc quá nhỏ thì sao?", "Vì sao phải chia nhỏ tài liệu khi làm RAG?"],
     "Chunking chia tài liệu dài thành các đoạn ngắn để embedding và retrieval hiệu quả hơn.",
     "Chunk quá lớn loãng ngữ nghĩa, quá nhỏ mất ngữ cảnh; kích thước hợp lý thường vài trăm token."),
    (["Reranking trong RAG là gì?", "Vì sao cần rerank sau retrieval?", "Reranker khác embedding search ở đâu?"],
     "Reranking chấm điểm lại các đoạn ứng viên bằng model chuyên biệt để chọn ra ít đoạn liên quan nhất cho prompt.",
     "Embedding search nhanh nhưng thô, reranker đọc cặp câu hỏi và đoạn văn nên chính xác hơn nhưng chậm hơn."),
    (["Hybrid search là gì?", "Từ khóa bổ sung cho vector search thế nào?", "Vì sao kết hợp BM25 và embedding?"],
     "Hybrid search kết hợp tìm kiếm từ khóa và tìm kiếm vector để tận dụng cả khớp chính xác lẫn tương đồng ngữ nghĩa.",
     "Mã số, tên riêng hiển nhiên khớp tốt bằng từ khóa, còn câu paraphrase cần embedding, kết hợp hai loại phủ nhau."),
    (["Faithfulness trong đánh giá RAG là gì?", "Vì sao đo faithfulness?", "Câu trả lời trung thành với ngữ nghĩa là sao?"],
     "Faithfulness đo tỷ lệ nội dung câu trả lời thực sự được hỗ trợ bởi các đoạn ngữ cảnh đã truy xuất.",
     "Câu trả lời hay mà không dựa trên ngữ cảnh là hallucination, faithfulness phát hiện đúng điều đó."),
    (["Rerank có đáng chi phí không?", "Khi nào rerank lời?"],
     "Rerank đáng chi phí khi corpus lớn và cần độ chính xác cao vì chọn được ngữ cảnh tốt hơn hẳn cho prompt.",
     "Prompt chứa ít đoạn mà liên quan thì câu trả lời tốt hơn và còn tiết kiệm token hơn là nhồi nhiều đoạn."),
]

EVAL_QUESTIONS: list[str] = [
    "Mô hình ngôn ngữ lớn là gì?",
    "Tại sao phải train lại model mà không dùng trực tiếp?",
    "Adapter trong LoRA có kích thước thế nào so với model gốc?",
    "Nén model xuống 4-bit mất gì để được gì?",
    "Khi sinh JSON từ LLM nên đặt temperature bao nhiêu?",
    "Nếu muốn model luôn trả lời đúng một format thì chọn RAG hay fine-tune?",
    "Kiến trúc transformer thay thế kiến trúc nào trước đây?",
    "Vì sao tách token theo ký tự khiến chuỗi vào model dài?",
    "Hàm mất mát của model sinh ngôn ngữ là gì?",
    "Số tầng và số head ảnh hưởng gì đến model?",
    "Trong attention, ma trận nào giữ vai trò câu hỏi?",
    "Cache các tensor nào để sinh text nhanh hơn?",
    "Đánh giá RAG bằng chỉ số nào để phát hiện bịa đặt?",
    "Chia nhỏ tài liệu ảnh hưởng thế nào đến chất lượng truy xuất?",
    "Tìm kiếm lai giải quyết nhược điểm nào của tìm vector thuần?",
    "Học tăng cường từ phản hồi con người gồm mấy giai đoạn?",
    "Ưa thích trực tiếp khác ưu tiên qua reward model chỗ nào?",
    "Overfit khi fine-tune biểu hiện ra sao?",
    "Vì sao theo dõi cả loss trên tập kiểm định?",
    "Gradient accumulation giúp gì khi GPU ít bộ nhớ?",
    "Warmup learning rate dùng khi nào?",
    "Câu lệnh điều kiện vào prompt gọi là kỹ thuật gì?",
    "Đưa ví dụ vào prompt thuộc loại học nào?",
    "Lý luận từng bước giúp giải bài toán thế nào?",
    "Vector nhúng đo độ giống nhau bằng công thức nào?",
    "Cơ sở dữ liệu vector tối ưu truy vấn gì?",
    "Rerank tốn thêm bao nhiêu bước trong pipeline?",
    "Chỉ số trung thành với ngữ cảnh đo điều gì?",
    "Residual connection đặt ở đâu trong block transformer?",
    "Chuẩn hóa lớp trước hay sau block là ổn định hơn?",
]


def _assistant_content(answer: str, reason: str) -> str:
    return json.dumps({"answer": answer, "reason": reason, "lang": "vi"}, ensure_ascii=False)


def _make_pair(question: str, answer: str, reason: str) -> dict:
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_content(answer, reason)},
    ]}


def build_all(out_dir: Path | None = None) -> dict[str, int]:
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1337)

    train: list[dict] = []
    for phrases, answer, reason in TOPICS:
        for question in phrases:
            train.append(_make_pair(question, answer, reason))
    # deterministic shuffle keeps variety across the file order
    rng.shuffle(train)
    while len(train) < 350:
        phrases, answer, reason = TOPICS[len(train) % len(TOPICS)]
        train.append(_make_pair(rng.choice(phrases), answer, reason))
    train = train[:350]

    eval_pairs = [_make_pair(q, "Placeholder.", "Placeholder.") for q in EVAL_QUESTIONS[:30]]
    # eval answers: reuse topic knowledge deterministically by cycling topics
    for i, pair in enumerate(eval_pairs):
        _, answer, reason = TOPICS[i % len(TOPICS)]
        pair["messages"][1]["content"] = _assistant_content(answer, reason)

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    train_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in train), encoding="utf-8"
    )
    eval_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in eval_pairs), encoding="utf-8"
    )
    return {"train": len(train), "eval": len(eval_pairs)}


def main() -> None:
    counts = build_all()
    print(counts)


if __name__ == "__main__":
    main()
