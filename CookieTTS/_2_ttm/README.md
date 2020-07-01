This is the Text-to-Mel folder. 

Inside here you will see two architectures.

---

### Tacotron2 has the following modifications available;

 - **Drop Frame Rate** taken from [MAXIMIZING MUTUAL INFORMATION FOR TACOTRON](https://arxiv.org/pdf/1909.01145.pdf)
	 - Claimed MOS change from **3.84** ->  **3.92**
 - **Global Style Tokens** from [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/pdf/1803.09017.pdf)
	 - Allows control over Style using a reference audio file
	 - No inference options are currently supplied for this
 - **TorchMoji Style Tokens**
	 - Use torchMoji to provde Style Tokens taken entirely from text.
	 - This is supported by most inference options since torchMoji uses text as the only input.
 - **Dynamic Convolution Attention** from [LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORM
SPEECH SYNTHESIS](https://arxiv.org/pdf/1910.10288.pdf)
	 - This implementation *works* (it allows long inputs) however stability is worse than Content-Location Hybrid Attention.
 - **GMM based Attention** from [LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORM
SPEECH SYNTHESIS](https://arxiv.org/pdf/1910.10288.pdf)
	 - This implementation *works* however struggles with very long pauses.

---

### Flow-TTS currently does not exist.

I intend to provide TorchMoji, speaking rate and emotion control options initially.