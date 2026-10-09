"""Read-only frozen Stage1 slots; no trainable representation encoder."""
import torch
from common.entity_inputs import EntityInputs

class EntityRepresentations:
    output_dim = 1024
    input_dims = None
    knowledge_bank = None
    def __init__(self, payload):
        self.slots, self.roles, self.counts = payload["slots"], payload["roles"], payload["counts"]
    def values(self, object_ids, topology):
        ids = object_ids.cpu().long()
        count = 2 if topology == "il" else 1 if topology == "molecule" else 0
        if count == 0 or not bool((self.counts[ids] == count).all()):
            raise ValueError("Frozen entity slot topology mismatch")
        return EntityInputs(self.slots[ids], self.roles[ids],
            torch.arange(2)[None, :].expand(len(ids), -1) < self.counts[ids, None])
