Đúng. Với cách bạn vừa định nghĩa thì mình đồng ý formulation nên được cố định như sau, và từ đây mình sẽ **chỉ dùng (\delta)** cho learnable weight update.

### Objective hiện tại

[
\boxed{
\delta P_{\mathrm{retain}} C_{\mathrm{erase}}
=============================================

W P_{\mathrm{target}}(-C_{\mathrm{erase}})
}
]

với

[
\boxed{
P_{\mathrm{target}}
===================

\operatorname{LowRankHighEnergy}
\left(
\operatorname{UnitNorm}(G)
\right)
}
]

Tức pipeline là

[
G
\rightarrow
\operatorname{UnitNorm}(G)
\rightarrow
\text{SVD}
\rightarrow
\text{top-}k
\rightarrow
P_{\mathrm{target}}.
]

Ở đây (P_{\mathrm{target}}) được học **chỉ từ geometry của raw target residuals (G)**. Đây đúng với construction hiện tại: normalize các pairwise residuals trước SVD rồi giữ leading singular directions. 

### Objective mới bạn đang đề xuất

[
\boxed{
\delta P_{\mathrm{retain}} C_{\mathrm{erase}}
=============================================

W P_{\mathrm{target}}^{\mathrm{new}}
(-C_{\mathrm{erase}})
}
]

với

[
\boxed{
P_{\mathrm{target}}^{\mathrm{new}}
==================================

\operatorname{LowRankHighEnergy}
\left(
\operatorname{UnitNorm}
\left(
G P_{\mathrm{retain}}
\right)
\right)
}
]

Pipeline trở thành

[
G
\rightarrow
GP_{\mathrm{retain}}
\rightarrow
\operatorname{UnitNorm}
\rightarrow
\text{SVD}
\rightarrow
\text{top-}k
\rightarrow
P_{\mathrm{target}}^{\mathrm{new}}.
]

**Đây đúng là thay đổi cốt lõi.**

Có một distinction rất quan trọng so với phần mình derive trước: ta **không project final residual một cách hậu nghiệm**. Ta project **raw observations (G) trước khi estimate target subspace**. Vì vậy (P_{\mathrm{retain}}) thay đổi chính cái geometry mà SVD nhìn thấy.

Và từ formulation này ta có một property rất đẹp. Vì

[
GP_{\mathrm{retain}}
]

nằm trong range của (P_{\mathrm{retain}}), nếu (P_{\mathrm{retain}}) là orthogonal projector thì các nonzero right singular vectors của (GP_{\mathrm{retain}}) cũng nằm trong range đó. Do vậy

[
\boxed{
\operatorname{range}
(P_{\mathrm{target}}^{\mathrm{new}})
\subseteq
\operatorname{range}
(P_{\mathrm{retain}})
}
]

và do đó

[
\boxed{
P_{\mathrm{retain}}
P_{\mathrm{target}}^{\mathrm{new}}
==================================

P_{\mathrm{target}}^{\mathrm{new}}
}
]

cũng như

[
\boxed{
P_{\mathrm{target}}^{\mathrm{new}}
P_{\mathrm{retain}}
===================

P_{\mathrm{target}}^{\mathrm{new}}.
}
]

Đây mới là property mà mình nghĩ nên dùng làm **starting point cho proof retention**.

Nó cho thấy objective mới có một symmetry/alignment mà objective cũ không đảm bảo:

[
\underbrace{\delta
P_{\mathrm{retain}}
C_{\mathrm{erase}}}_{\text{input side: retain-low geometry}}
============================================================

\underbrace{
W P_{\mathrm{target}}^{\mathrm{new}}
(-C_{\mathrm{erase}})
}_{\text{target side: cũng nằm trong retain-low geometry}}.
]

Trong objective cũ,

[
P_{\mathrm{target}}
===================

\operatorname{LowRankHighEnergy}(\operatorname{UnitNorm}(G)),
]

nói chung **không có**

[
P_{\mathrm{retain}}P_{\mathrm{target}}
======================================

P_{\mathrm{target}}.
]

Nên hai phía đang được xây từ **hai geometry khác nhau**.

Còn objective mới enforce:

[
\boxed{
\operatorname{range}(P_{\mathrm{target}}^{\mathrm{new}})
\subseteq
\operatorname{range}(P_{\mathrm{retain}})
}
]

ngay tại bước xây target subspace.

**Chỗ này chính là chìa khóa.** Nếu mình derive lại proof từ đầu dựa **chính xác hai objectives bạn vừa viết**, mình nghĩ có thể đi xa hơn câu trả lời trước và chỉ ra bằng decomposition (P_{\mathrm{retain}}+(I-P_{\mathrm{retain}})=I) rằng objective cũ chứa một target component mà phía (\delta P_{\mathrm{retain}}C_{\mathrm{erase}}) không được thiết kế để represent, còn objective mới loại component đó *trước khi rank-(k) budget được phân bổ*. Đây sẽ là argument toán học sát Figure 1 hơn.
