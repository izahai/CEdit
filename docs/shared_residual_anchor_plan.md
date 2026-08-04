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
  `shared_residual_max_norm person`, và `legacy null`.
- Sample cùng seed để đánh giá khả năng xóa identity và mức ảnh hưởng tới retain concepts.

## Lưu ý

Virtual anchor không nhất thiết tương ứng với một prompt hợp lệ. Với mode mới, từng target không được map chính xác về `"person"`; thay vào đó, tất cả target được dịch cùng một hướng và centroid của virtual anchors được căn với reference anchor.
