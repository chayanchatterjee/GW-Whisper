# GW-Whisper

We introduce GW-Whisper, an innovative application of [OpenAI](https://openai.com/)’s [Whisper model](https://arxiv.org/abs/2212.04356), originally designed for speech recognition, to [gravitational wave (GW)](https://www.ligo.caltech.edu/page/what-are-gw) data analysis. As the volume of data from advanced detectors like [LIGO](https://en.wikipedia.org/wiki/LIGO) and [Virgo](https://www.virgo-gw.eu/) grows, traditional methods face scalability challenges. GW-Whisper leverages Whisper’s pre-trained architecture to address critical tasks in GW astronomy, including signal detection and glitch classification, by fine-tuning the model using the parameter-efficient [DoRA](https://arxiv.org/abs/2402.09353) method. This fine-tuning updates only 0.5% of the model’s parameters, significantly reducing computational costs.
The architecture of GW-Whisper is shown ![below](https://github.com/chayanchatterjee/GW-Whisper/blob/main/imgs/Figure_1.png):

