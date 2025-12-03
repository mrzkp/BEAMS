import numpy as np
from scipy import constants

"""
Used to generate the dataset for NN and SMO.
"""


# System parameters as given in Table I
SBS_DEN = 1e-4         # λS: 1 × 10−4m−2
SBS_POWER_DB = 20      # PS: 20 dBm
PATHS = 2              # L: 2 propagation paths
NUM_MT_ANT = 2         # NMT: 2 MT antennas
NUM_SBS_ANT = 32       # NSBS: 32 SBS antennas
RADIUS = 100           # R: 100m radius
CB_BEAMS = 8           # NC: 8 candidate vectors

class MmWaveDatasetGenerator:
    def __init__(self):
        self.sbs_power_db = SBS_POWER_DB
        self.num_paths = PATHS           
        self.num_mt_antennas = NUM_MT_ANT
        self.num_sbs_antennas = NUM_SBS_ANT
        self.radius = RADIUS              
        self.num_codebook_beams = CB_BEAMS
        
        self.sbs_power = 10 ** ((self.sbs_power_db - 30) / 10)  # Convert dBm to Watts
        
        self.num_sbs = int(np.floor(self.sbs_density * np.pi * self.radius**2))
        
        self.frequency = 28e9              # 28 GHz (mmWave frequency)
        self.wavelength = constants.c / self.frequency
        self.k = 2 * np.pi / self.wavelength
        self.d_mt = self.wavelength / 2    # Half-wavelength antenna spacing
        self.d_sbs = self.wavelength / 2
        
        self.codebook = self._generate_codebook()
        self.num_samples = 30_000
        
    """
    Generate DFT-based codebook with NC=8 candidate vectors
    """
    def _generate_codebook(self):
        codebook = np.zeros((self.num_sbs_antennas, self.num_codebook_beams), dtype=complex)
        angles = np.linspace(-np.pi/2, np.pi/2, self.num_codebook_beams, endpoint=False)
        
        for i, angle in enumerate(angles):
            array_response = np.exp(1j * self.k * self.d_sbs * 
                                  np.arange(self.num_sbs_antennas) * np.sin(angle))
            codebook[:, i] = array_response / np.sqrt(self.num_sbs_antennas)
        
        return codebook

    """
    Generate channel matrix H_S,k based on Saleh-Valenzuela model
    """
    def _generate_channel(self):
        aoa = np.random.uniform(-np.pi/2, np.pi/2, self.num_paths)
        aod = np.random.uniform(-np.pi/2, np.pi/2, self.num_paths)
        
        alpha = (np.random.normal(0, 1/np.sqrt(2), self.num_paths) + 
                1j * np.random.normal(0, 1/np.sqrt(2), self.num_paths))
        
        a_mt = np.zeros((self.num_mt_antennas, self.num_paths), dtype=complex)
        a_sbs = np.zeros((self.num_sbs_antennas, self.num_paths), dtype=complex)
        
        for l in range(self.num_paths):
            a_mt[:, l] = np.exp(1j * self.k * self.d_mt * 
                               np.arange(self.num_mt_antennas) * np.sin(aoa[l])) / \
                               np.sqrt(self.num_mt_antennas)
            
            a_sbs[:, l] = np.exp(1j * self.k * self.d_sbs * 
                                np.arange(self.num_sbs_antennas) * np.sin(aod[l])) / \
                                np.sqrt(self.num_sbs_antennas)
        
        gamma = np.sqrt(self.num_sbs_antennas * self.num_mt_antennas / self.num_paths)
        H = gamma * sum(alpha[l] * np.outer(a_mt[:, l], a_sbs[:, l].conj()) 
                       for l in range(self.num_paths))
        
        return H, aoa, aod, alpha

    def _calculate_snr(self, H, beam):
        noise_power = 1e-13
        H_reshaped = H.reshape(self.num_mt_antennas, self.num_sbs_antennas)
        received_signal = H_reshaped @ beam
        received_power = np.sum(np.abs(received_signal) ** 2) * self.sbs_power
        return received_power / noise_power

    def _find_optimal_beam(self, H):
        snr_values = np.zeros(self.num_codebook_beams)
        
        for i in range(self.num_codebook_beams):
            beam = self.codebook[:, i].reshape(-1, 1)
            snr_values[i] = self._calculate_snr(H, beam)
            
        return np.argmax(snr_values)

    def generate_dataset(self):
        features = []
        labels = []
        
        for _ in range(self.num_samples):
            H, aoa, aod, alpha = self._generate_channel()
            
            feature = np.concatenate([
                aoa.real,              # AoA for each path
                aod.real,              # AoD for each path
                alpha.real,            # Real part of path gains
                alpha.imag,            # Imaginary part of path gains
                H.reshape(-1).real,    # Real part of channel matrix
                H.reshape(-1).imag     # Imaginary part of channel matrix
            ])
            
            optimal_beam = self._find_optimal_beam(H)
            
            features.append(feature)
            labels.append(optimal_beam)
        
        return np.array(features), np.array(labels)

def main():
    generator = MmWaveDatasetGenerator() # 30_000 TS
    features, labels = generator.generate_dataset()
    
    print("DEBUGGING")
    print(f"Dataset generated:")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    
    np.save('Naive_SVM/mmwave_dataset.npy', {'features': features, 'labels': labels})

if __name__ == "__main__":
    main()
