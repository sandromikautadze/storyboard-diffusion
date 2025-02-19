# Data

The raw dataset is introduced in  

> Anyi Rao, Jiaze Wang, Linning Xu, Xuekun Jiang, Qingqiu Huang, Bolei Zhou, Dahua Lin
> *A Unified Framework for Shot Type Classification Based on Subject Centric Lens*
> **In ECCV, 2020**.

It contains movie shots from publicly available movie trailers. Each clip is labeled with five shot scale categories (ECS, CS, MS, FS, LS) and four shot movement categories (static, motion, push, pull).

Our functions are in `src/data.py` and the process is done in `lens_classifier.ipynb` in the **Data Extraction** section. In short, we select the middle frame of each clip and save only the shot scale label. We end up with roughly 30k labeled images. The results are completely reproducible.

PS: For memory reasons, the datasets are not pushed to the repository. 