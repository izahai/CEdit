# Experiments

## Few-concept erasure

Concept-erasure results for Stable Diffusion v1.4. **CS** denotes CLIP Score and **FID** denotes Fréchet Inception Distance. ↓ means lower is better; ↑ means higher is better. Bold values are the best result in each column. Non-target and MS-COCO columns measure preservation.

## Instance concepts

### Original model

| Model | Snoopy CS | Mickey CS | Spongebob CS | Pikachu CS | Hello Kitty CS | MS-COCO CS |
|:--|--:|--:|--:|--:|--:|--:|
| SD v1.4 | 28.51 | 26.62 | 27.30 | 27.44 | 27.77 | 26.53 |

### Erase Snoopy

| Method | Snoopy CS ↓ | Mickey FID ↓ | Spongebob FID ↓ | Pikachu FID ↓ | Hello Kitty FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 25.44 | 37.08 | 38.92 | 26.14 | 36.52 | 26.40 | 21.20 |
| MACE | 20.90 | 105.97 | 102.77 | 65.71 | 75.42 | 26.09 | 42.62 |
| RECE | **18.38** | 26.63 | 34.42 | 21.99 | 32.35 | 26.39 | 25.61 |
| UCE | 23.19 | 24.87 | 29.86 | 19.06 | 27.86 | 26.46 | 22.18 |
| SPEED | 23.50 | **23.41** | 24.64 | 16.81 | 21.74 | 26.48 | 19.95 |
| Ours | 22.63 | 24.03 | **24.05** | **16.47** | **20.73** | **26.53** | **18.84** |

### Erase Snoopy and Mickey

| Method | Snoopy CS ↓ | Mickey CS ↓ | Spongebob FID ↓ | Pikachu FID ↓ | Hello Kitty FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 25.26 | 26.58 | 45.08 | 35.57 | 41.48 | 26.42 | 24.34 |
| MACE | 20.53 | 20.63 | 112.01 | 91.72 | 106.88 | 25.50 | 55.15 |
| RECE | **18.57** | **19.14** | 35.85 | 26.05 | 40.77 | 26.31 | 30.30 |
| UCE | 23.60 | 24.79 | 30.58 | 23.51 | 31.76 | 26.38 | 26.06 |
| SPEED | 23.58 | 23.62 | 29.67 | 22.51 | 28.23 | 26.47 | 23.66 |
| Ours | 23.14 | 21.47 | **27.90** | **21.67** | **27.08** | **26.49** | **23.19** |

### Erase Snoopy, Mickey, and Spongebob

| Method | Snoopy CS ↓ | Mickey CS ↓ | Spongebob CS ↓ | Pikachu FID ↓ | Hello Kitty FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 24.92 | 26.46 | 25.12 | 46.47 | 48.24 | 26.37 | 26.71 |
| MACE | 19.86 | 19.35 | 20.12 | 110.12 | 128.56 | 23.39 | 66.39 |
| RECE | **18.17** | **18.87** | **16.23** | 40.52 | 52.06 | 26.32 | 32.51 |
| UCE | 23.29 | 24.63 | 19.08 | 29.20 | 38.15 | 26.30 | 28.71 |
| SPEED | 23.69 | 23.93 | 21.39 | **21.40** | **26.22** | **26.51** | 24.99 |
| Ours | 23.09 | 21.54 | 20.18 | 24.01 | 26.78 | 26.50 | **24.63** |

## Artistic styles

### Original model

| Model | Van Gogh CS | Picasso CS | Monet CS | Paul Gauguin CS | Caravaggio CS | MS-COCO CS |
|:--|--:|--:|--:|--:|--:|--:|
| SD v1.4 | 28.75 | 27.98 | 28.91 | 29.80 | 26.27 | 26.53 |

### Erase Van Gogh

| Method | Van Gogh CS ↓ | Picasso FID ↓ | Monet FID ↓ | Paul Gauguin FID ↓ | Caravaggio FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 28.16 | 77.01 | 63.80 | 63.20 | 79.25 | 26.46 | 18.36 |
| MACE | 26.66 | 69.92 | 60.88 | 56.18 | 69.04 | 26.50 | 23.15 |
| RECE | 26.39 | 60.57 | 61.09 | 47.07 | 72.85 | 26.52 | 23.54 |
| UCE | 28.10 | 43.02 | 40.49 | 32.62 | 61.72 | 26.54 | 19.63 |
| SPEED | 26.29 | **35.86** | **16.85** | 24.94 | **39.75** | **26.55** | 20.36 |
| Ours | **25.12** | 48.80 | 21.28 | **24.01** | 54.70 | 26.52 | **17.85** |

### Erase Picasso

| Method | Van Gogh FID ↓ | Picasso CS ↓ | Monet FID ↓ | Paul Gauguin FID ↓ | Caravaggio FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 60.44 | 26.97 | 36.23 | 65.23 | 79.12 | 26.43 | 20.02 |
| MACE | 59.58 | 26.48 | 37.02 | 46.35 | 66.20 | 26.47 | 22.86 |
| RECE | 51.09 | 26.66 | 25.39 | 46.08 | 75.61 | 26.48 | 23.03 |
| UCE | 37.58 | 26.99 | 16.72 | 32.48 | 59.27 | 26.50 | 20.33 |
| SPEED | **19.18** | **26.22** | 19.87 | **24.73** | **43.63** | **26.51** | 19.98 |
| Ours | 32.42 | 26.56 | **16.00** | 29.03 | 49.80 | 26.45 | **17.32** |

### Erase Monet

| Method | Van Gogh FID ↓ | Picasso FID ↓ | Monet CS ↓ | Paul Gauguin FID ↓ | Caravaggio FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ConAbl | 68.77 | 64.25 | 27.05 | 57.33 | 71.88 | 26.45 | 21.03 |
| MACE | 61.50 | 48.41 | 25.98 | 49.66 | 65.87 | 26.47 | 22.76 |
| RECE | 56.26 | 45.97 | 25.87 | 46.38 | 64.19 | 26.49 | 24.94 |
| UCE | 42.25 | 38.73 | 27.12 | 33.00 | 56.49 | **26.51** | 21.58 |
| SPEED | 28.78 | 41.21 | 25.06 | 27.85 | 55.20 | 26.48 | 20.87 |
| Ours | **21.89** | **34.40** | **24.56** | **18.32** | **39.39** | 26.43 | **20.26** |
