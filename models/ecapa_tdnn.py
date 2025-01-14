import torch
import torch.nn as nn
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

class MainModel(nn.Module):
    def __init__(self, nOut=192, **kwargs):
        super(MainModel, self).__init__()
        
        self.model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Note: ECAPA-TDNN from SpeechBrain outputs 192-dimensional embeddings
        self.embedding_size = nOut
        
    def forward(self, x):
        # Add channel dimension if missing
        if len(x.shape) == 2:
            # x shape is [batch_size, num_samples]
            x = x.unsqueeze(1)  # Convert to [batch_size, 1, num_samples]
        
        # Ensure the input is float32
        x = x.float()
        
        # Handle the case where input might be [num_samples] 
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, num_samples]
            
        with torch.no_grad():
            # Convert the numpy array output to a torch tensor
            embeddings = torch.from_numpy(self.model(x)).to(x.device)
            
        return embeddings
