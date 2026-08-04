# Shared-Residual Anchor Plan

## Mục tiêu

Với các target embedding `t_i`, tạo virtual anchor embedding `a_i` sao cho:

```text
a_i - t_i = r
```

với cùng một residual `r` cho mọi target. Khi đó statistic dùng để edit model là:

```text
mean((a_i - t_i)^T t_i) = r^T mean(t_i)
```

nên có rank tối đa bằng 1.

## Thay đổi dự kiến

1. Thêm `--anchor_mode` vào `train_erase_null.py`:
   - `legacy`: giữ nguyên cách tính hiện tại.
   - `shared_residual_mean`: sử dụng virtual anchor embedding dựa trên mean embedding.
   - `shared_residual_max_norm`: chọn residual có L2 norm lớn nhất và dùng nó cho mọi target.
   - `shared_residual_sign_aligned`: dùng shared residual mean và đảo dấu riêng cho
     từng cặp để cùng hướng với residual gốc.
   - `shared_residual_cosine_medoid`: chọn residual gốc có mean cosine similarity
     cao nhất với các residual còn lại.
   - `shared_residual_abs_cosine_medoid`: chọn residual gốc có mean absolute cosine
     similarity cao nhất với các residual còn lại.
   - `shared_residual_smallest_cosine_medoid`: chọn residual gốc có mean cosine
     similarity thấp nhất với các residual còn lại.
2. Encode tất cả target và reference anchor trước khi tính statistics.
3. Tính residual chung:

   ```text
   r = mean(anchor_embeddings) - mean(target_embeddings)
   ```

   Với một anchor như `"person"`, công thức trở thành `r = e("person") - mean(t_i)`.
4. Tạo virtual anchor cho từng target bằng `a_i = t_i + r`.
5. Giữ nguyên retain projection và công thức cập nhật weight của SPEED.

Với `shared_residual_max_norm`, residual chung được chọn theo:

```text
r_i = anchor_i - target_i
k = argmax_i ||r_i||_2
r = r_k
```

Nếu embedding gồm nhiều token, `||r_i||_2` được tính trên embedding đã flatten,
tương đương Frobenius norm của ma trận embedding.

Với `shared_residual_sign_aligned`, dấu của residual cho cặp thứ `i` được chọn theo:

```text
r_shared = mean(anchor_i - target_i)
sign_i = +1.0  if dot(anchor_i - target_i, r_shared) >= 0
sign_i = -1.0  otherwise
r_i = sign_i * r_shared
```

Dot product dùng Frobenius inner product khi embedding có nhiều token. Khi chạy,
chương trình in số cặp nhận dấu `+1.0` và `-1.0`.

Với hai mode cosine medoid, residual được flatten và chuẩn hóa trước khi tạo cosine
similarity matrix. Điểm trên đường chéo bị loại khỏi mean score khi có nhiều hơn một
target:

```text
score_i = mean_{j != i}(cosine(r_i, r_j))
```

Mode `shared_residual_abs_cosine_medoid` thay `cosine` bằng `abs(cosine)`. Residual
gốc có score cao nhất được dùng chung cho mọi target; hai mode này không tự động đảo
dấu residual cho từng cặp.

Mode `shared_residual_smallest_cosine_medoid` dùng cùng cosine score nhưng chọn
`argmin` thay vì `argmax`, vì vậy nó ưu tiên residual ít tương đồng nhất với phần còn
lại của tập. Residual được chọn vẫn giữ nguyên norm gốc trước khi áp dụng
`residual_scale`.

Các mode chọn một residual gốc (`shared_residual_max_norm` và hai cosine-medoid
mode) sẽ log index và prompt target cung cấp shared residual. Mode
`shared_residual_sign_aligned` dùng residual mean nên log rõ rằng không có một target
prompt nguồn duy nhất. `legacy` và `shared_residual_mean` không in source log này.

Có thể điều chỉnh độ lớn residual cho mọi mode bằng:

```text
r_scaled = residual_scale * r
```

CLI sử dụng `--residual_scale`, mặc định là `1.0` và yêu cầu một số hữu hạn lớn hơn
0. Residual được scale sau khi mode đã tạo residual riêng/chung, nhưng trước khi tính
edit statistic và diagnostics.

## Kiểm chứng

- Kiểm tra mọi `a_i - t_i` bằng nhau trong sai số floating-point.
- In singular values/rank của residual matrix và edit statistic.
- Smoke test với 10 celebrities, so sánh `legacy person`, `shared_residual_mean person`,
  `shared_residual_max_norm person`, `shared_residual_sign_aligned person`, và
  `legacy null`.
- Sample cùng seed để đánh giá khả năng xóa identity và mức ảnh hưởng tới retain concepts.

## Lưu ý

Virtual anchor không nhất thiết tương ứng với một prompt hợp lệ. Với mode mới, từng target không được map chính xác về `"person"`; thay vào đó, tất cả target được dịch cùng một hướng và centroid của virtual anchors được căn với reference anchor.
