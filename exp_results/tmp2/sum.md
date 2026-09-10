\section{EXPERIMENTS}

\subsection{ON FEW-CONCEPT ERASURE}

\begin{table*}[t]
\centering
\caption{Concept erasure results. CS denotes CLIP Score and FID denotes
Fr\'echet Inception Distance. $\downarrow$ indicates lower is better,
while $\uparrow$ indicates higher is better. Our method consistently
improves prior preservation for non-target concepts and general concepts
from MS-COCO (shaded in \protect\colorbox{fidblue}{blue}), while achieving effective concept
erasure.}
\label{tab:concept_erasure}

\vspace{2mm}
\scriptsize
\setlength{\tabcolsep}{3.8pt}
\renewcommand{\arraystretch}{1.08}

\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}c@{\hspace{0.01\textwidth}}c@{}}

% =========================================================
% LEFT TABLE
% =========================================================
\begin{tabular}{l|ccccc|cc}
\toprule
\textbf{Concept}
&
\textbf{\textit{Snoopy}}
&
\textbf{\textit{Mickey}}
&
\textbf{\textit{Spongebob}}
&
\textbf{\textit{Pikachu}}
&
\textbf{\textit{Hello Kitty}}
&
\multicolumn{2}{c}{\textbf{MS-COCO}}
\\

&
CS & CS & CS & CS & CS & CS & FID
\\
\midrule

SD v1.4
& 28.51 & 26.62 & 27.30 & 27.44 & 27.77
& 26.53 & -- \\

\midrule
\multicolumn{8}{c}{Erase \textbf{\textit{Snoopy}}} \\
\midrule

&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 25.44 & 37.08 & 38.92 & 26.14 & 36.52
& 26.40 & 21.20 \\

MACE
& 20.90 & 105.97 & 102.77 & 65.71 & 75.42
& 26.09 & 42.62 \\

RECE
& \textbf{18.38} & 26.63 & 34.42 & 21.99 & 32.35
& 26.39 & 25.61 \\

UCE
& 23.19 & 24.87 & 29.86 & 19.06 & 27.86
& 26.46 & 22.18 \\

SPEED
& 23.50
& \textbf{23.41}
& 24.64
& 16.81
& 21.74
& 26.48
& 19.95
\\

\midrule
Ours
& 22.63 
& 24.03 & \textbf{24.05} 
& \textbf{16.47} & \textbf{20.73} 
& \textbf{26.53} & \textbf{18.84} \\

\midrule
\multicolumn{8}{c}{
Erase \textbf{\textit{Snoopy}} and \textbf{\textit{Mickey}}
} \\
\midrule

&
CS$\downarrow$
&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 25.26 & 26.58 & 45.08 & 35.57 & 41.48
& 26.42 & 24.34 \\

MACE
& 20.53 & 20.63 & 112.01 & 91.72 & 106.88
& 25.50 & 55.15 \\

RECE
& \textbf{18.57}
& \textbf{19.14}
& 35.85 & 26.05 & 40.77
& 26.31 & 30.30 \\

UCE
& 23.60 & 24.79 & 30.58 & 23.51 & 31.76
& 26.38 & 26.06 \\

SPEED
& 23.58
& 23.62
& 29.67
& 22.51
& 28.23
& 26.47
& 23.66
\\

\midrule
Ours
& 23.14 & 21.47 & \textbf{27.90} & \textbf{21.67} & \textbf{27.08 }
& \textbf{26.49} & \textbf{23.19} \\

\midrule
\multicolumn{8}{c}{
Erase \textbf{\textit{Snoopy}},
\textbf{\textit{Mickey}}, and
\textbf{\textit{Spongebob}}
} \\
\midrule

&
CS$\downarrow$
&
CS$\downarrow$
&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 24.92 & 26.46 & 25.12 & 46.47 & 48.24
& 26.37 & 26.71 \\

MACE
& 19.86 & 19.35 & 20.12 & 110.12 & 128.56
& 23.39 & 66.39 \\

RECE
& \textbf{18.17}
& \textbf{18.87}
& \textbf{16.23}
& 40.52 & 52.06
& 26.32 & 32.51 \\

UCE
& 23.29 & 24.63 & 19.08 & 29.20 & 38.15
& 26.30 & 28.71 \\

SPEED
& 23.69
& 23.93
& 21.39
& \textbf{21.40}
& \textbf{26.22}
& \textbf{26.51}
& 24.99
\\

\midrule
Ours
& 23.09 & 21.54 & 20.18 & 24.01 & 26.78 
& 26.50 & \textbf{24.63} \\
\bottomrule
\end{tabular}

&

% =========================================================
% RIGHT TABLE
% =========================================================
\begin{tabular}{l|ccccc|cc}
\toprule
\textbf{Concept}
&
\textbf{\textit{Van Gogh}}
&
\textbf{\textit{Picasso}}
&
\textbf{\textit{Monet}}
&
\textbf{\textit{Paul Gauguin}}
&
\textbf{\textit{Caravaggio}}
&
\multicolumn{2}{c}{\textbf{MS-COCO}}
\\

&
CS & CS & CS & CS & CS & CS & FID
\\
\midrule

SD v1.4
& 28.75 & 27.98 & 28.91 & 29.80 & 26.27
& 26.53 & -- \\

\midrule
\multicolumn{8}{c}{Erase \textbf{\textit{Van Gogh}}} \\
\midrule

&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 28.16 & 77.01 & 63.80 & 63.20 & 79.25
& 26.46 & 18.36 \\

MACE
& 26.66 & 69.92 & 60.88 & 56.18 & 69.04
& 26.50 & 23.15 \\

RECE
& 26.39 & 60.57 & 61.09 & 47.07 & 72.85
& 26.52 & 23.54 \\

UCE
& 28.10 & 43.02 & 40.49 & 32.62 & 61.72
& 26.54 & 19.63 \\

SPEED
& 26.29
& \textbf{35.86}
& \textbf{16.85}
& 24.94
& \textbf{39.75}
& \textbf{26.55}
& 20.36
\\

\midrule
Ours
& \textbf{25.12} & 48.80 & 21.28 & \textbf{24.01} & 54.70
& 26.52 & \textbf{17.85} \\

\midrule
\multicolumn{8}{c}{Erase \textbf{\textit{Picasso}}} \\
\midrule

&
\cellcolor{fidblue}FID$\downarrow$
&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 60.44 & 26.97 & 36.23 & 65.23 & 79.12
& 26.43 & 20.02 \\

MACE
& 59.58 & 26.48 & 37.02 & 46.35 & 66.20
& 26.47 & 22.86 \\

RECE
& 51.09 & 26.66 & 25.39 & 46.08 & 75.61
& 26.48 & 23.03 \\

UCE
& 37.58 & 26.99 & 16.72 & 32.48 & 59.27
& 26.50 & 20.33 \\

SPEED
& \textbf{19.18}
& \textbf{26.22}
& 19.87
& \textbf{24.73}
& \textbf{43.63}
& \textbf{26.51}
& 19.98
\\

\midrule
Ours
& 32.42 & 26.56 & \textbf{16.00} & 29.03 & 49.80
& 26.45 & \textbf{17.32} \\

\midrule
\multicolumn{8}{c}{Erase \textbf{\textit{Monet}}} \\
\midrule

&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
CS$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}FID$\downarrow$
&
\cellcolor{fidblue}CS$\uparrow$
&
\cellcolor{fidblue}FID$\downarrow$
\\
\midrule

ConAbl
& 68.77 & 64.25 & 27.05 & 57.33 & 71.88
& 26.45 & 21.03 \\

MACE
& 61.50 & 48.41 & 25.98 & 49.66 & 65.87
& 26.47 & 22.76 \\

RECE
& 56.26 & 45.97 & 25.87 & 46.38 & 64.19
& 26.49 & 24.94 \\

UCE
& 42.25 & 38.73 & 27.12 & 33.00 & 56.49
& \textbf{26.51} & 21.58 \\

SPEED
& 28.78
& 41.21
& 25.06
& 27.85
& 55.20
& 26.48
& 20.87
\\

\midrule
Ours
& \textbf{21.89} & \textbf{34.40} & \textbf{24.56}
& \textbf{18.32} & \textbf{39.39} & 26.43 & \textbf{20.26} \\
\bottomrule
\end{tabular}

\end{tabular}
}

\vspace{-2mm}
\end{table*}
