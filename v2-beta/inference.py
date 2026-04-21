from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image


CATEGORY_NAMES: List[str] = [
    "Trojan",         
    "Worm",           
    "Downloader",     
    "Backdoor_RAT",   
    "Ransomware",     
    "Adware_PUA",    
    "Spyware_Stealer",
    "Virus",         
    "Botnet",         
    "Other",        
]
CATEGORY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORY_NAMES)}
N_CATEGORIES: int = len(CATEGORY_NAMES)

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

    "USB_propagation", "email_propagation", "P2P_propagation", "IM_propagation",

    "anti_debugging", "anti_vm_detection", "sandbox_evasion",

    "DDoS_capability", "IRC_communication", "domain_generation_algorithm",
    "proxy_usage", "spam_sending",

    "cryptocurrency_mining", "screenshot_capture", "browser_hooking",
    "click_fraud", "rootkit_behavior", "bootkit_behavior",
    "screen_locking", "shadow_copy_deletion", "fake_alert_display",
    "self_deletion", "brute_force_attack", "remote_shell", "modular_architecture",
]
N_PREDICATES: int = len(ALL_PREDICATES)
PREDICATE_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(ALL_PREDICATES)}

ANCHOR_FAMILIES = {
    "Allaple.A", "Allaple.L", "Yuner.A", "Agent.FYI", "Rbot!gen",
    "Adialer.C", "Instantaccess", "Dialplatform.B",
    "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3", "Obfuscator.AD",
    "C2LOP.gen!g", "Swizzor.gen!E",
    "zbot", "gandcrab", "smokeloader", "mira", "bladabindi", "berbew",
    "mydoom", "vobfus", "shifu", "blocker", "fakeav", "upatre",
    "glupteba", "wabot", "coinminer", "tofsee", "nitol", "padodor",
}


PREDICATE_LABELS: Dict[str, str] = {
    "has_network_activity":      "Network Activity",
    "contacts_c2_server":        "C2 Server Contact",
    "downloads_payload":         "Downloads Payload",
    "exfiltrates_data":          "Data Exfiltration",
    "port_scanning":             "Port Scanning",
    "dns_tunneling":             "DNS Tunneling",
    "modifies_system_files":     "Modifies System Files",
    "creates_files":             "Creates Files",
    "deletes_files":             "Deletes Files",
    "hides_files":               "Hides Files",
    "encrypts_files":            "Encrypts Files",
    "process_injection":         "Process Injection",
    "creates_processes":         "Creates Processes",
    "terminates_processes":      "Terminates Processes",
    "elevates_privileges":       "Privilege Escalation",
    "disables_security":         "Disables Security Tools",
    "has_packing":               "Binary Packing",
    "has_obfuscation":           "Code Obfuscation",
    "has_encryption":            "Uses Encryption",
    "polymorphic_code":          "Polymorphic Code",
    "large_binary_size":         "Large Binary Size",
    "shows_advertising":         "Displays Advertising",
    "dials_premium_numbers":     "Dials Premium Numbers",
    "steals_credentials":        "Credential Theft",
    "logs_keystrokes":           "Keylogging",
    "self_replication":          "Self-Replication",
    "network_propagation":       "Network Propagation",
    "persistence_mechanisms":    "Persistence Mechanisms",
    "registry_modification":     "Registry Modification",
    "USB_propagation":           "USB Drive Propagation",
    "email_propagation":         "Email Propagation",
    "P2P_propagation":           "P2P Network Propagation",
    "IM_propagation":            "Instant Messenger Spread",
    "anti_debugging":            "Anti-Debugging",
    "anti_vm_detection":         "Anti-VM Detection",
    "sandbox_evasion":           "Sandbox Evasion",
    "DDoS_capability":           "DDoS Capability",
    "IRC_communication":         "IRC Communication",
    "domain_generation_algorithm": "Domain Generation (DGA)",
    "proxy_usage":               "Proxy Usage",
    "spam_sending":              "Spam Sending",
    "cryptocurrency_mining":     "Cryptocurrency Mining",
    "screenshot_capture":        "Screenshot Capture",
    "browser_hooking":           "Browser Hooking",
    "click_fraud":               "Click Fraud",
    "rootkit_behavior":          "Rootkit Behavior",
    "bootkit_behavior":          "Bootkit Behavior",
    "screen_locking":            "Screen Locking",
    "shadow_copy_deletion":      "Shadow Copy Deletion",
    "fake_alert_display":        "Fake Alert Display",
    "self_deletion":             "Self-Deletion",
    "brute_force_attack":        "Brute Force Attack",
    "remote_shell":              "Remote Shell",
    "modular_architecture":      "Modular Architecture",
}


class _Stage1BinaryCNN(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class _ResNet50Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x).flatten(1)


class _HierarchicalHeads(nn.Module):

    def __init__(self, feature_dim: int, n_categories: int, n_families: int):
        super().__init__()
        self.n_categories = n_categories

        self.category_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_categories),
        )
        self.family_head = nn.Sequential(
            nn.Linear(feature_dim + n_categories, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_families),
        )

    def forward(self, features: torch.Tensor):
        cat_logits = self.category_head(features)
        cat_probs  = F.softmax(cat_logits, dim=1)
        fam_logits = self.family_head(torch.cat([features, cat_probs], dim=1))
        return cat_logits, fam_logits


class _SymbolicPredicateGrounding(nn.Module):

    def __init__(self, feature_dim: int, n_predicates: int):
        super().__init__()

        self.predicate_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_predicates)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([ext(features) for ext in self.predicate_extractors], dim=1)



class _KnowledgeBase:

    def __init__(
        self,
        predicate_names: List[str],
        category_names: List[str],
        family_names: List[str],
    ):
        self._pred = {p: i for i, p in enumerate(predicate_names)}
        self._cat  = {c: i for i, c in enumerate(category_names)}
        self._fam  = {f: i for i, f in enumerate(family_names)}
        self.n_cat = len(category_names)
        self.n_fam = len(family_names)
        self.cat_rules: List[Dict] = []
        self.fam_rules: List[Dict] = []

    def _add(self, store, ants, con_idx, w, rule_type):
        idxs = [self._pred[a] for a in ants if a in self._pred]
        if idxs and con_idx is not None:
            store.append({"ants": idxs, "con": con_idx, "w": w, "type": rule_type})

    def add_cat(self, ants, category, w=1.0, rule_type='and'):
        self._add(self.cat_rules, ants, self._cat.get(category), w, rule_type)

    def add_fam(self, ants, family, w=1.0, rule_type='and'):
        self._add(self.fam_rules, ants, self._fam.get(family), w, rule_type)

    def _apply(self, preds: torch.Tensor, rules, n_cls: int) -> torch.Tensor:
        B, device = preds.size(0), preds.device
        buckets: List[List[torch.Tensor]] = [[] for _ in range(n_cls)]
        for r in rules:
            if r["type"] == 'and':
                act = torch.prod(preds[:, r["ants"]], dim=1) * r["w"]
            else: 
                act = (1 - torch.prod(1 - preds[:, r["ants"]], dim=1)) * r["w"]
            buckets[r["con"]].append(act)
        cols = []
        for b in buckets:
            cols.append(torch.stack(b).max(0)[0] if b else torch.zeros(B, device=device))
        return torch.stack(cols, dim=1)

    def reason(self, preds: torch.Tensor):
        return (self._apply(preds, self.cat_rules, self.n_cat),
                self._apply(preds, self.fam_rules, self.n_fam))


def _build_kb(predicate_names, category_names, family_names) -> _KnowledgeBase:

    kb = _KnowledgeBase(predicate_names, category_names, family_names)

  
  
    kb.add_cat(["process_injection", "persistence_mechanisms"],                  "Trojan",          0.82)
    kb.add_cat(["registry_modification", "has_obfuscation", "persistence_mechanisms"], "Trojan",   0.80)
    kb.add_cat(["disables_security", "modifies_system_files"],                   "Trojan",          0.72)
    kb.add_cat(["process_injection", "registry_modification"],                   "Trojan",          0.78)

    kb.add_cat(["self_replication", "network_propagation"],                      "Worm",            0.90)
    kb.add_cat(["self_replication", "USB_propagation"],                          "Worm",            0.85)
    kb.add_cat(["self_replication", "registry_modification", "persistence_mechanisms"], "Worm",     0.88)
    kb.add_cat(["network_propagation", "USB_propagation",
                "email_propagation", "P2P_propagation"],                         "Worm",            0.80, 'or')

    kb.add_cat(["encrypts_files", "shadow_copy_deletion"],                       "Ransomware",      0.95)
    kb.add_cat(["encrypts_files"],                                                "Ransomware",      0.80)
    kb.add_cat(["screen_locking"],                                                "Ransomware",      0.75)
  
    kb.add_cat(["steals_credentials", "exfiltrates_data"],                       "Spyware_Stealer", 0.90)
    kb.add_cat(["steals_credentials", "browser_hooking"],                        "Spyware_Stealer", 0.88)
    kb.add_cat(["logs_keystrokes", "exfiltrates_data"],                          "Spyware_Stealer", 0.82)
    kb.add_cat(["logs_keystrokes", "hides_files"],                               "Spyware_Stealer", 0.80)
    kb.add_cat(["screenshot_capture", "exfiltrates_data"],                       "Spyware_Stealer", 0.82)
    kb.add_cat(["steals_credentials", "anti_vm_detection"],                      "Spyware_Stealer", 0.78)

    kb.add_cat(["DDoS_capability"],                                              "Botnet",          0.85)
    kb.add_cat(["spam_sending", "has_network_activity"],                         "Botnet",          0.80)
    kb.add_cat(["cryptocurrency_mining"],                                         "Botnet",          0.78)
   
    kb.add_cat(["shows_advertising", "has_network_activity"],                    "Adware_PUA",      0.88)
    kb.add_cat(["click_fraud"],                                                   "Adware_PUA",      0.75)

    kb.add_cat(["downloads_payload", "contacts_c2_server"],                      "Downloader",      0.78)
    kb.add_cat(["downloads_payload", "anti_debugging"],                          "Downloader",      0.75)
   
    kb.add_cat(["contacts_c2_server", "remote_shell"],                           "Backdoor_RAT",    0.88)
    kb.add_cat(["contacts_c2_server", "persistence_mechanisms",
                "exfiltrates_data"],                                             "Backdoor_RAT",    0.82)
  
    kb.add_cat(["self_replication", "modifies_system_files"],                    "Virus",           0.75)
    kb.add_cat(["self_replication", "modifies_system_files", "has_obfuscation"], "Virus",           0.83)
    kb.add_cat(["polymorphic_code", "self_replication"],                         "Virus",           0.87)

  
    kb.add_fam(["self_replication", "network_propagation"],                      "Allaple.A",       0.93)
    kb.add_fam(["self_replication", "network_propagation", "port_scanning"],     "Allaple.L",       0.93)
    kb.add_fam(["self_replication", "network_propagation", "persistence_mechanisms"], "Yuner.A",    0.92)
    kb.add_fam(["self_replication", "network_propagation", "registry_modification"],  "Yuner.A",    0.88)
    kb.add_fam(["self_replication", "DDoS_capability", "brute_force_attack"],    "mira",            0.95)
    kb.add_fam(["self_replication", "email_propagation", "DDoS_capability"],     "mydoom",          0.90)
    kb.add_fam(["USB_propagation", "downloads_payload", "hides_files"],          "vobfus",          0.92)
    kb.add_fam(["self_replication", "P2P_propagation", "IRC_communication"],     "wabot",           0.88)
    kb.add_fam(["encrypts_files", "shadow_copy_deletion", "contacts_c2_server"], "gandcrab",        0.93)
    kb.add_fam(["screen_locking", "persistence_mechanisms"],                     "blocker",         0.90)
    kb.add_fam(["steals_credentials", "browser_hooking",
                "process_injection", "domain_generation_algorithm"],             "zbot",            0.92)
    kb.add_fam(["browser_hooking", "steals_credentials", "exfiltrates_data"],    "berbew",          0.92)
    kb.add_fam(["steals_credentials", "anti_debugging",
                "browser_hooking", "anti_vm_detection"],                         "shifu",           0.90)
    kb.add_fam(["steals_credentials", "exfiltrates_data"],                       "Lolyda.AA1",      0.88)
    kb.add_fam(["steals_credentials", "logs_keystrokes"],                        "Lolyda.AA2",      0.85)
    kb.add_fam(["steals_credentials", "exfiltrates_data"],                       "Lolyda.AA3",      0.86)
    kb.add_fam(["steals_credentials", "browser_hooking", "polymorphic_code"],    "padodor",         0.90)
    kb.add_fam(["anti_debugging", "anti_vm_detection",
                "sandbox_evasion", "downloads_payload"],                         "smokeloader",     0.95)
    kb.add_fam(["downloads_payload", "process_injection", "self_deletion"],      "upatre",          0.90)
    kb.add_fam(["has_obfuscation", "hides_files", "disables_security"],          "Obfuscator.AD",   0.94)
    kb.add_fam(["downloads_payload", "has_network_activity"],                    "Swizzor.gen!E",   0.85)
    kb.add_fam(["contacts_c2_server", "downloads_payload"],                      "Agent.FYI",       0.88)
    kb.add_fam(["process_injection", "contacts_c2_server"],                      "Rbot!gen",        0.90)
    kb.add_fam(["contacts_c2_server", "elevates_privileges", "persistence_mechanisms"], "C2LOP.gen!g", 0.90)
    kb.add_fam(["contacts_c2_server", "logs_keystrokes", "remote_shell"],        "bladabindi",      0.90)
    kb.add_fam(["cryptocurrency_mining", "rootkit_behavior", "proxy_usage"],     "glupteba",        0.88)
    kb.add_fam(["cryptocurrency_mining", "creates_processes"],                   "coinminer",       0.88)
    kb.add_fam(["spam_sending", "DDoS_capability", "modular_architecture"],      "tofsee",          0.88)
    kb.add_fam(["DDoS_capability", "has_network_activity"],                      "nitol",           0.85)
    kb.add_fam(["shows_advertising", "has_network_activity"],                    "Adialer.C",       0.93)
    kb.add_fam(["shows_advertising", "downloads_payload"],                       "Instantaccess",   0.91)
    kb.add_fam(["dials_premium_numbers", "has_network_activity"],                "Dialplatform.B",  0.95)
    kb.add_fam(["fake_alert_display", "disables_security"],                      "fakeav",          0.92)

    return kb



_TRANSFORM_STAGE1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


_TRANSFORM_STAGE23 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


BENIGN_CONF_GATE: float = 0.65

SUSPICIOUS_FAMILY_THRESH: float = 0.30


class NeSyWareInference:

    WEIGHTS_DIR = Path(__file__).parent / "weights"

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._stage1_model:  Optional[_Stage1BinaryCNN]     = None
        self._cnn:           Optional[_ResNet50Backbone]    = None
        self._heads:         Optional[_HierarchicalHeads]   = None
        self._grounding:     Optional[_SymbolicPredicateGrounding] = None
        self._kb:            Optional[_KnowledgeBase]       = None
        self._family_names:  Optional[List[str]]            = None
        self._fam_to_cat:    Optional[Dict[int, int]]       = None
        self._models_loaded: bool                           = False

    def load(self, progress_callback=None) -> None:
        def _notify(msg: str):
            if progress_callback:
                progress_callback(msg)

        _notify("Caricamento Stage 1 — CNN binaria...")
        self._load_stage1()

        _notify("Caricamento Stage 2/3 — NeSyWare ResNet-50 + LTN...")
        self._load_stage23()

        self._models_loaded = True
        _notify("Modelli caricati. Sistema pronto.")

    def _load_stage1(self):
        path = self.WEIGHTS_DIR / "stage1_binary.pth"
        ck = torch.load(path, map_location=self.device, weights_only=False)

        model = _Stage1BinaryCNN()

        if isinstance(ck, dict) and "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"])
        else:

            try:
                model.load_state_dict(ck)
            except Exception:

                state = {k.replace("module.", ""): v for k, v in ck.items()
                         if k not in ("epoch", "optimizer_state_dict",
                                      "val_auc", "val_loss", "val_acc",
                                      "val_f1", "val_prec", "val_rec", "lr")}
                model.load_state_dict(state, strict=False)

        model.to(self.device).eval()
        self._stage1_model = model

    def _load_stage23(self):
        path = self.WEIGHTS_DIR / "stage23_nesyware.pth"
        ck = torch.load(path, map_location=self.device, weights_only=False)


        family_names:       List[str]       = ck["family_names"]
        fam_to_cat:         Dict[int, int]  = ck["family_idx_to_category_idx"]
        n_families = len(family_names)


        cnn = _ResNet50Backbone()
        cnn.load_state_dict(ck["cnn_state_dict"])
        cnn.to(self.device).eval()

        heads = _HierarchicalHeads(cnn.feature_dim, N_CATEGORIES, n_families)
        heads.load_state_dict(ck["heads_state_dict"])
        heads.to(self.device).eval()


        grounding = _SymbolicPredicateGrounding(cnn.feature_dim, N_PREDICATES)
        grounding.load_state_dict(ck["grounding_state_dict"])
        grounding.to(self.device).eval()


        kb = _build_kb(ALL_PREDICATES, CATEGORY_NAMES, family_names)

        self._cnn          = cnn
        self._heads        = heads
        self._grounding    = grounding
        self._kb           = kb
        self._family_names = family_names
        self._fam_to_cat   = fam_to_cat

    def analyze(self, filepath: str) -> Dict:

        if not self._models_loaded:
            raise RuntimeError("Chiamare load() prima di analyze().")

        result = {
            "is_malware":         False,
            "stage1_confidence":  0.0,
            "stage1_level":       "uncertain",
            "is_uncertain":       True,

            "suspicious":         False,
            "suspicious_reason":  "",
            "category":           None,
            "category_conf":      None,
            "top5_categories":    [],
            "all_category_probs": [],
            "family":               None,
            "family_conf":          None,
            "is_anchor_family":     False,
            "family_profile":       [],  
            "family_profile_label": "",  
            "top5_families":        [],   
            "active_predicates":  [],
            "all_predicates":     [],
            "error":              None,
        }

        try:
            from pe_to_image import pe_to_rgb_crops, pe_to_rgb_crops_mw

            crops_std = pe_to_rgb_crops(filepath,    img_size=224)  
            crops_mw  = pe_to_rgb_crops_mw(filepath, img_size=224) 
            all_crops = crops_std + crops_mw

            crop_probs = []
            with torch.no_grad():
                for crop_img in all_crops:
                    x = _TRANSFORM_STAGE1(crop_img).unsqueeze(0).to(self.device)
                    p = torch.sigmoid(self._stage1_model(x)).item()
                    crop_probs.append(p)

            n_votes  = sum(1 for p in crop_probs if p >= 0.5)
            is_mw    = n_votes >= 6   
            aligned  = [p for p in crop_probs if (p >= 0.5) == is_mw]
            avg_prob = float(np.mean(aligned)) if aligned else float(np.mean(crop_probs))
            conf     = avg_prob if is_mw else (1.0 - avg_prob)

            result["is_malware"]        = is_mw
            result["stage1_confidence"] = round(conf, 4)

   
            if conf >= 0.80:
                level = 'high'
            elif conf >= 0.60:
                level = 'medium'
            else:
                level = 'uncertain'
            result["stage1_level"]   = level
            result["is_uncertain"]   = (level == 'uncertain')


            from pe_to_image import pe_to_rgb_crops
            img_s23 = pe_to_rgb_crops(filepath, img_size=224)[0]  
            x2 = _TRANSFORM_STAGE23(img_s23).unsqueeze(0).to(self.device)

            ALPHA_NEURAL   = 0.65   
            ALPHA_SYMBOLIC = 0.35   

            with torch.no_grad():
                features   = self._cnn(x2)
                predicates = self._grounding(features)
                cat_logits, fam_logits = self._heads(features)
                cat_kb, fam_kb = self._kb.reason(predicates)

                combined_cat = (ALPHA_NEURAL   * cat_logits +
                                ALPHA_SYMBOLIC * torch.log(cat_kb + 1e-10))
                combined_fam = (ALPHA_NEURAL   * fam_logits +
                                ALPHA_SYMBOLIC * torch.log(fam_kb + 1e-10))

                cat_probs = F.softmax(combined_cat, dim=1)[0]
                fam_probs = F.softmax(combined_fam, dim=1)[0]
                pred_vals = predicates[0]

  
            cat_idx  = int(cat_probs.argmax())
            cat_conf = float(cat_probs[cat_idx])
            result["category"]      = CATEGORY_NAMES[cat_idx]
            result["category_conf"] = round(cat_conf, 4)

            all_cat = sorted(
                [(i, CATEGORY_NAMES[i], round(float(cat_probs[i]), 4))
                 for i in range(N_CATEGORIES)],
                key=lambda x: x[2], reverse=True,
            )
            result["all_category_probs"] = [(n, p) for _, n, p in all_cat]


            fam_idx  = int(fam_probs.argmax())
            fam_conf = float(fam_probs[fam_idx])
            fam_name = self._family_names[fam_idx]
            result["family"]           = fam_name
            result["family_conf"]      = round(fam_conf, 4)
            result["is_anchor_family"] = fam_name in ANCHOR_FAMILIES


            if not is_mw and fam_conf >= SUSPICIOUS_FAMILY_THRESH:
                result["suspicious"] = True
                s1_low = conf < BENIGN_CONF_GATE
                gate_note = (
                    f"Stage 1 low-confidence benign ({conf:.0%}) + " if s1_low
                    else f"Stage 1 benign ({conf:.0%}) overridden by "
                )
                result["suspicious_reason"] = (
                    f"{gate_note}Stage 2/3 family '{fam_name}' "
                    f"[{CATEGORY_NAMES[cat_idx]}] at {fam_conf:.0%} confidence"
                )

            fam_cat_idx = self._fam_to_cat.get(fam_idx, -1)

            top5_cat = []
            for ci, cname, cprob in all_cat[:5]:
                tag = "✓" if ci == fam_cat_idx else "✗"
                top5_cat.append((cname, cprob, tag))
            result["top5_categories"] = top5_cat


            FAMILY_THRESHOLD   = 0.08   
            MIN_REPORT_CONF    = 0.20 
            MAX_FAMILIES       = 8      
            all_fam_sorted = sorted(
                range(len(self._family_names)),
                key=lambda fi: float(fam_probs[fi]), reverse=True
            )
            family_profile = []
            for fi in all_fam_sorted:
                fconf = round(float(fam_probs[fi]), 4)
                if fconf < FAMILY_THRESHOLD:
                    break
                if len(family_profile) >= MAX_FAMILIES:
                    break
                fname = self._family_names[fi]
                tag   = "✓" if self._fam_to_cat.get(fi, -1) == cat_idx else "✗"
                family_profile.append((fname, fconf, tag))


            top1_conf = family_profile[0][1] if family_profile else 0.0
            if top1_conf >= 0.70:
                profile_label = "high_confidence"
            elif top1_conf >= 0.50:
                profile_label = "moderate"
            elif top1_conf >= MIN_REPORT_CONF:
                profile_label = "ambiguous"
            else:

                profile_label = "inconclusive"
                family_profile = []   

            if profile_label == "inconclusive":
                result["family"]           = None
                result["family_conf"]      = round(fam_conf, 4)
                result["is_anchor_family"] = False
            else:
                result["family"]           = fam_name
                result["family_conf"]      = round(fam_conf, 4)
                result["is_anchor_family"] = fam_name in ANCHOR_FAMILIES
            result["family_profile"]       = family_profile
            result["family_profile_label"] = profile_label
            result["top5_families"]        = family_profile[:5]  


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
