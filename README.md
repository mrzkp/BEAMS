# B.E.A.M.S: Beamforming and Enhanced Algorithms for Millimeter-Wave Systems

Millimeter-wave (mmWave) small cell networks are critical components of 5G wireless communication systems. By deploying a dense network of mmWave small cell base stations (SBSs), these systems support thousands of connections and deliver high transmission rates for a wide range of localized services. SBSs enable short-range communications with mobile terminals (MTs), minimizing signal propagation loss. Leveraging mmWave technology, multiple SBSs can employ large antenna arrays to form directional analog beams, facilitating concurrent transmissions to MTs. However, as the number of SBSs and MTs increases, traditional signal processing methods struggle to maintain performance efficiency.

In the original research, the authors propose a three-step methodology:

1. **Random Distribution Modeling**: The spatial distribution of SBSs is modeled using a heterogeneous Poisson point process (HPPP). This approach enables the calculation of the average sum rate (ASR) for MTs under concurrent transmission scenarios.

2. **Machine Learning for Beam Selection**: A comprehensive database of downlink SBS conditions is established for machine learning training. An iterative support vector machine (SVM) classifier is developed to optimize analog beam selection for each SBS.

3. **Iterative SMO Algorithm**: An iterative sequential minimal optimization (SMO) training algorithm is introduced, enabling SBSs to perform efficient and low-complexity analog beam selection during concurrent transmissions.

This methodology demonstrates faster performance compared to traditional channel estimation algorithms.

For our project, we aim to develop a novel machine learning-based approach for concurrent transmission in mmWave small cell networks. We will evaluate and compare the performance of various methods to identify optimal strategies for enhancing system efficiency.
