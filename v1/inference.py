from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

MALIMG_FAMILIES: List[str] = [
    "Adialer.C",       
    "Agent.FYI",        
    "Allaple.A",        
    "Allaple.L",        
    "Alueron.gen!J",   
    "C2LOP.P",          
    "C2LOP.gen!g",     
    "Dialplatform.B",   
    "Dontovo.A",       
    "Fakerean",         
    "Instantaccess",   
    "Lolyda.AA1",    
    "Lolyda.AA2",       
    "Lolyda.AA3",      
    "Lolyda.AT",       
    "Malex.gen!J",      
    "Obfuscator.AD",   
    "Rbot!gen",        
    "Skintrim.N",       
    "Swizzor.gen!E",   
    "Swizzor.gen!I",    
    "VB.AT",           
    "Wintrim.BX",       
    "Yuner.A",          
]
N_FAMILIES: int = len(MALIMG_FAMILIES)
FAMILY_TO_IDX: Dict[str, int] = {f: i for i, f in enumerate(MALIMG_FAMILIES)}


FAMILY_CATEGORY: Dict[str, str] = {
    "Adialer.C":      "Adware/Dialer",
    "Agent.FYI":      "Trojan",
    "Allaple.A":      "Worm",
    "Allaple.L":      "Worm",
    "Alueron.gen!J":  "Trojan",
    "C2LOP.P":        "Backdoor",
    "C2LOP.gen!g":    "Backdoor",
    "Dialplatform.B": "Dialer",
    "Dontovo.A":      "Trojan",
    "Fakerean":       "Rogue/FakeAV",
    "Instantaccess":  "Adware",
    "Lolyda.AA1":     "Password Stealer",
    "Lolyda.AA2":     "Password Stealer",
    "Lolyda.AA3":     "Password Stealer",
    "Lolyda.AT":      "Password Stealer",
    "Malex.gen!J":    "Generic Malware",
    "Obfuscator.AD":  "Obfuscator",
    "Rbot!gen":       "Trojan/Bot",
    "Skintrim.N":     "Trojan",
    "Swizzor.gen!E":  "Downloader",
    "Swizzor.gen!I":  "Downloader",
    "VB.AT":          "Generic Malware",
    "Wintrim.BX":     "Trojan",
    "Yuner.A":        "Worm",
}


ALL_PREDICATES: List[str] = [
    "has_network_activity", "contacts_c2_server", "downloads_payload",
    "exfiltrates_data", "port_scanning", "dns_tunneling",
    "modifies_system_files", "creates_files", "deletes_files",
    "hides_files", "encrypts_files",
    "process_injection", "creates_processes", "terminates_processes",
    "elevates_privileges", "disables_security",
    "has_packing", "has_obfuscation", "has_encryption",
    "polymorphic_code", "large_binary_size",
    "shows_advertising", "dials_premium_numbers", "steals_credentials",
    "logs_keystrokes", "self_replication", "network_propagation",
    "persistence_mechanisms", "registry_modification",
]
N_PREDICATES: int = len(ALL_PREDICATES)

PREDICATE_LABELS: Dict[str, str] = {
    "has_network_activity":   "Network Activity",
    "contacts_c2_server":     "C2 Server Contact",
    "downloads_payload":      "Downloads Payload",
    "exfiltrates_data":       "Data Exfiltration",
    "port_scanning":          "Port Scanning",
    "dns_tunneling":          "DNS Tunneling",
    "modifies_system_files":  "Modifies System Files",
    "creates_files":          "Creates Files",
    "deletes_files":          "Deletes Files",
    "hides_files":            "Hides Files",
    "encrypts_files":         "Encrypts Files",
    "process_injection":      "Process Injection",
    "creates_processes":      "Creates Processes",
    "terminates_processes":   "Terminates Processes",
    "elevates_privileges":    "Privilege Escalation",
    "disables_security":      "Disables Security Tools",
    "has_packing":            "Binary Packing",
    "has_obfuscation":        "Code Obfuscation",
    "has_encryption":         "Uses Encryption",
    "polymorphic_code":       "Polymorphic Code",
    "large_binary_size":      "Large Binary Size",
    "shows_advertising":      "Displays Advertising",
    "dials_premium_numbers":  "Dials Premium Numbers",
    "steals_credentials":     "Credential Theft",
    "logs_keystrokes":        "Keylogging",
    "self_replication":       "Self-Replication",
    "network_propagation":    "Network Propagation",
    "persistence_mechanisms": "Persistence Mechanisms",
    "registry_modification":  "Registry Modification",
}


ANCHOR_FAMILIES = {
    "Allaple.A", "Allaple.L", "Yuner.A",
    "Adialer.C", "Dialplatform.B", "Instantaccess",
    "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3",
    "Obfuscator.AD", "C2LOP.gen!g",
}

class _MalImgCNN(nn.Module):


    def __init__(self, num_classes: int = N_FAMILIES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.feature_dim = 512 * 4 * 4  

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )


class _SymbolicPredicateGrounding(nn.Module):
 

    def __init__(self, feature_dim: int = 8192, n_predicates: int = N_PREDICATES):
        super().__init__()
        self.predicate_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),          nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1),            nn.Sigmoid(),
            )
            for _ in range(n_predicates)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([ext(features) for ext in self.predicate_extractors], dim=1)



class _KnowledgeBase:


    def __init__(self):
        pred_idx = {p: i for i, p in enumerate(ALL_PREDICATES)}
        fam_idx  = FAMILY_TO_IDX

        self._rules: List[Dict] = []

        def add(ants: List[str], fam: str, w: float, rtype: str = "and"):
            idxs = [pred_idx[a] for a in ants if a in pred_idx]
            fi   = fam_idx.get(fam)
            if idxs and fi is not None:
                self._rules.append({"ants": idxs, "fam": fi, "w": w, "type": rtype})


        add(["self_replication", "network_propagation"],        "Allaple.A",  0.95)
        add(["self_replication", "network_propagation"],        "Allaple.L",  0.95)
        add(["self_replication", "network_propagation"],        "Yuner.A",    0.90)
  
        add(["contacts_c2_server", "downloads_payload"],        "Agent.FYI",  0.90)
        add(["process_injection",  "contacts_c2_server"],       "Rbot!gen",   0.92)

        add(["shows_advertising",  "has_network_activity"],     "Adialer.C",  0.93)
        add(["shows_advertising",  "downloads_payload"],        "Instantaccess", 0.91)
        add(["dials_premium_numbers", "has_network_activity"],  "Dialplatform.B", 0.95)

        add(["steals_credentials", "exfiltrates_data"],         "Lolyda.AA1", 0.90)
        add(["steals_credentials", "logs_keystrokes"],          "Lolyda.AA2", 0.88)
        add(["steals_credentials", "exfiltrates_data"],         "Lolyda.AA3", 0.89)

        add(["has_obfuscation", "hides_files", "disables_security"], "Obfuscator.AD", 0.94)

        add(["contacts_c2_server", "elevates_privileges", "persistence_mechanisms"], "C2LOP.gen!g", 0.92)
        add(["contacts_c2_server", "creates_processes"],        "C2LOP.P",    0.90)

        add(["downloads_payload",  "has_network_activity"],     "Swizzor.gen!E", 0.88)
        add(["downloads_payload",  "has_network_activity"],     "Swizzor.gen!I", 0.88)

    def reason(self, preds: torch.Tensor) -> torch.Tensor:

        B, device = preds.size(0), preds.device
        buckets: List[List[torch.Tensor]] = [[] for _ in range(N_FAMILIES)]

        for r in self._rules:
            if r["type"] == "and":
                act = torch.prod(preds[:, r["ants"]], dim=1) * r["w"]
            else:
                act = (1 - torch.prod(1 - preds[:, r["ants"]], dim=1)) * r["w"]
            buckets[r["fam"]].append(act)

        cols = [
            torch.stack(b).max(0)[0] if b else torch.zeros(B, device=device)
            for b in buckets
        ]
        return torch.stack(cols, dim=1)



_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def _load_image(filepath: str) -> Image.Image:

    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
        img = Image.open(filepath).convert("L")
    else:
    
        from pe_to_image import pe_to_binary_image
        img_rgb = pe_to_binary_image(filepath, img_size=128)
        img = img_rgb.convert("L")

    return img


class NeSyWareInference:


    WEIGHTS_DIR = Path(__file__).parent / "weights"
    ALPHA_NEURAL    = 0.7
    ALPHA_SYMBOLIC  = 0.3

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cnn:        Optional[_MalImgCNN]                 = None
        self._grounding:  Optional[_SymbolicPredicateGrounding] = None
        self._kb:         Optional[_KnowledgeBase]              = None
        self._loaded:     bool                                  = False

    def load(self, progress_callback=None) -> None:
        def _notify(msg: str):
            if progress_callback:
                progress_callback(msg)

        _notify("Loading MalImg-24 NeSyWare model...")
        path = self.WEIGHTS_DIR / "malimg_nesyware.pth"
        ck   = torch.load(path, map_location=self.device, weights_only=False)

        cnn = _MalImgCNN(num_classes=N_FAMILIES)
        cnn.load_state_dict(ck["cnn_state_dict"])
        cnn.to(self.device).eval()

        grounding = _SymbolicPredicateGrounding()
        grounding.load_state_dict(ck["grounding_state_dict"])
        grounding.to(self.device).eval()

        self._cnn       = cnn
        self._grounding = grounding
        self._kb        = _KnowledgeBase()
        self._loaded    = True
        _notify("Model loaded. System ready.")


    def analyze(self, filepath: str) -> Dict:

        if not self._loaded:
            raise RuntimeError("Call load() before analyze().")

        result: Dict = {
            "family":               None,
            "family_conf":          0.0,
            "is_anchor_family":     False,
            "category":             None,
            "family_profile":       [],
            "family_profile_label": "inconclusive",
            "top5_families":        [],
            "active_predicates":    [],
            "all_predicates":       [],
            "error":                None,
        }

        try:
            img  = _load_image(filepath)
            x    = _TRANSFORM(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features   = self._cnn.features(x).flatten(1) 
                predicates = self._grounding(features)            
                neural_logits = self._cnn.classifier(features)  
                kb_scores     = self._kb.reason(predicates)      

                sym_logits    = torch.log(kb_scores + 1e-10)
                combined      = (self.ALPHA_NEURAL   * neural_logits +
                                 self.ALPHA_SYMBOLIC * sym_logits)
                probs         = F.softmax(combined, dim=1)[0]    
                pred_vals     = predicates[0]                   

    
            fam_idx  = int(probs.argmax())
            fam_conf = float(probs[fam_idx])
            fam_name = MALIMG_FAMILIES[fam_idx]

  
            FAMILY_THRESH  = 0.08
            MIN_CONF       = 0.20
            MAX_SHOW       = 8

            sorted_fi = sorted(range(N_FAMILIES),
                               key=lambda fi: float(probs[fi]), reverse=True)
            profile = []
            for fi in sorted_fi:
                fc = round(float(probs[fi]), 4)
                if fc < FAMILY_THRESH or len(profile) >= MAX_SHOW:
                    break
                profile.append((MALIMG_FAMILIES[fi], fc))

            top1 = profile[0][1] if profile else 0.0
            if top1 >= 0.70:
                plabel = "high_confidence"
            elif top1 >= 0.50:
                plabel = "moderate"
            elif top1 >= MIN_CONF:
                plabel = "ambiguous"
            else:
                plabel = "inconclusive"
                profile = []

            result["family"]               = fam_name if plabel != "inconclusive" else None
            result["family_conf"]          = round(fam_conf, 4)
            result["is_anchor_family"]     = fam_name in ANCHOR_FAMILIES
            result["category"]             = FAMILY_CATEGORY.get(fam_name, "Unknown")
            result["family_profile"]       = profile
            result["family_profile_label"] = plabel
            result["top5_families"]        = profile[:5]


            all_preds = sorted(
                [(ALL_PREDICATES[i], round(float(pred_vals[i]), 4))
                 for i in range(N_PREDICATES)],
                key=lambda x: x[1], reverse=True,
            )
            result["all_predicates"]    = all_preds
            result["active_predicates"] = [(p, v) for p, v in all_preds if v >= 0.40]

        except Exception as exc:
            import traceback
            result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        return result
