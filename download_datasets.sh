# Downloads 2D maze datasets used in our ICML 2018 paper.

OUTPUT_DIR=./mazes

mkdir ${OUTPUT_DIR}

# 15x15 mazes (NEWS)
wget https://cmu.box.com/shared/static/l46uqg78f2iik8avr9rdfyfi28vyek7o.npz -O ${OUTPUT_DIR}/m15_news_10k.npz
wget https://cmu.box.com/shared/static/voqgj886o1gfx7ievbytzybonlu8s56n.npz -O ${OUTPUT_DIR}/m15_news_25k.npz
wget https://cmu.box.com/shared/static/4bp5xxs7ilfohyosy941sc8bi9fbed8w.npz -O ${OUTPUT_DIR}/m15_news_100k.npz

# 15x15 mazes (Moore)
wget https://cmu.box.com/shared/static/p8aanmo5kj1bm9949njmigntmbol0o7s.npz -O ${OUTPUT_DIR}/m15_moore_10k.npz
wget https://cmu.box.com/shared/static/1nmxgma8uvnuezfifqxieliiy9qrx4pe.npz -O ${OUTPUT_DIR}/m15_moore_25k.npz
wget https://cmu.box.com/shared/static/gg78z5ka2sjx9jcbj6m31du1v9vmfmhm.npz -O ${OUTPUT_DIR}/m15_moore_100k.npz

# 15x15 mazes (DiffDrive)
wget https://cmu.box.com/shared/static/3rv3aghi8df17vwnidcj1kd0qci2qa94.npz -O ${OUTPUT_DIR}/m15_diffdrive_10k.npz
wget https://cmu.box.com/shared/static/3rv3aghi8df17vwnidcj1kd0qci2qa94.npz -O ${OUTPUT_DIR}/m15_diffdrive_25k.npz
wget https://cmu.box.com/shared/static/pjfw2rwj88ibx4ako6balz8d810qujw1.npz -O ${OUTPUT_DIR}/m15_diffdrive_100k.npz

# 28x28 mazes
wget https://cmu.box.com/shared/static/2bat54yisnnx5yybzl5uhtzkex0g65sx.npz -O ${OUTPUT_DIR}/m28_news_25k.npz
wget https://cmu.box.com/shared/static/c6st11gqtx1n1xs1lu86sys00yf7yu84.npz -O ${OUTPUT_DIR}/m28_moore_25k.npz
wget https://cmu.box.com/shared/static/r3bgykf8zss8ro0pfqw8sc3pgjfi1urg.npz -O ${OUTPUT_DIR}/m28_diffdrive_25k.npz
