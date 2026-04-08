import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import librosa
import soundfile as sf
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from pathlib import Path
from io import BytesIO
import json
import logging
import uuid
import hashlib

logger = logging.getLogger(__name__)

IMAGE_MODEL_PATH  = Path("models/image_bird_classifier.pth")
IMAGE_LABELS_PATH = Path("models/image_class_labels.json")
AUDIO_MODEL_PATH  = Path("models/audio_bird_classifier.pth")
AUDIO_LABELS_PATH = Path("models/audio_class_labels.json")
METADATA_PATH     = Path("data/raw/species_metadata.json")


class BioSyncModel:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️  Using device: {self.device}")

        self.is_loaded          = False
        self.species_list       = []
        self.species_metadata   = {}
        self.idx_to_class       = {}
        self.audio_idx_to_class = {}

        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)

        self._load_species_data()
        self._load_image_classifier()
        self._load_audio_classifier()

        self.image_gen = None
        self._try_load_image_generator()

        self.is_loaded = True
        logger.info("✅ BioSync model fully loaded")


    # ──────────────────────────────────────────────────────────────────────────
    # LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def _load_species_data(self):
        logger.info("📂 Loading species data...")

        if IMAGE_LABELS_PATH.exists():
            with open(IMAGE_LABELS_PATH, "r") as f:
                raw = json.load(f)
            self.idx_to_class = {int(k): v for k, v in raw.items()}
            self.species_list = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
            logger.info(f"✅ Loaded {len(self.species_list)} image class labels")
        else:
            logger.warning("⚠️  image_class_labels.json not found!")
            self.species_list = ["Unknown_Bird"]
            self.idx_to_class = {0: "Unknown_Bird"}

        if AUDIO_LABELS_PATH.exists():
            with open(AUDIO_LABELS_PATH, "r") as f:
                raw = json.load(f)
            self.audio_idx_to_class = {int(k): v for k, v in raw.items()}
            logger.info(f"✅ Loaded {len(self.audio_idx_to_class)} audio class labels")
        else:
            logger.warning("⚠️  audio_class_labels.json not found — using image labels")
            self.audio_idx_to_class = self.idx_to_class

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.species_metadata = json.load(f)
        else:
            self.species_metadata = {}

    def _load_image_classifier(self):
        logger.info("🖼️  Loading VGG16 image classifier...")
        num_classes = len(self.species_list)

        self.vgg16 = tv_models.vgg16(weights=None)

        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg16.features[17:].parameters():
            param.requires_grad = True

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

        if IMAGE_MODEL_PATH.exists():
            checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=self.device)
            self.vgg16.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc", 0)
            logger.info(f"✅ Loaded VGG16 (Val Acc: {val_acc:.1f}%)")
        else:
            logger.warning("⚠️  No trained image model found!")

        self.vgg16 = self.vgg16.to(self.device)
        self.vgg16.eval()

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        logger.info("✅ Image classifier ready")

    def _load_audio_classifier(self):
        logger.info("🎵 Loading CNN-LSTM audio classifier...")

        self.audio_sr = 22050
        self.duration = 5
        self.n_mels   = 128
        self.n_fft    = 2048
        self.hop_len  = 512
        self.fixed_t  = 216

        class CNNBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(0.2)
                )
            def forward(self, x):
                return self.block(x)

        class BirdCNNLSTM(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.cnn = nn.Sequential(
                    CNNBlock(1,   32),
                    CNNBlock(32,  64),
                    CNNBlock(64,  128),
                    CNNBlock(128, 256),
                )
                self.lstm = nn.LSTM(
                    input_size=256 * 8,
                    hidden_size=512,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.4,
                    bidirectional=True
                )
                self.attention = nn.Linear(1024, 1)
                self.bn        = nn.BatchNorm1d(1024)
                self.drop      = nn.Dropout(0.5)
                self.fc        = nn.Linear(1024, num_classes)

            def forward(self, x):
                x = self.cnn(x)
                B, C, H, T = x.shape
                x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)
                out, _ = self.lstm(x)
                attn = torch.softmax(self.attention(out), dim=1)
                ctx  = (attn * out).sum(dim=1)
                ctx  = self.bn(ctx)
                ctx  = self.drop(ctx)
                return self.fc(ctx)

        num_classes = len(self.audio_idx_to_class)
        self.audio_classifier = BirdCNNLSTM(num_classes).to(self.device)

        if AUDIO_MODEL_PATH.exists():
            checkpoint = torch.load(AUDIO_MODEL_PATH, map_location=self.device)
            self.audio_classifier.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc", 0)
            logger.info(f"✅ Loaded CNN-LSTM audio model (Val Acc: {val_acc:.1f}%)")
        else:
            logger.warning("⚠️  No trained audio model found — random weights!")

        self.audio_classifier.eval()
        logger.info("✅ Audio classifier ready")

    def _try_load_image_generator(self):
        try:
            from diffusers import StableDiffusionPipeline
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.image_gen = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1", torch_dtype=dtype
            ).to(self.device)
            logger.info("✅ Stable Diffusion loaded")
        except Exception as e:
            self.image_gen = None
            logger.warning(f"⚠️  Stable Diffusion not loaded: {e}")


    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def image_to_audio(self, image_path: str) -> dict:
        logger.info(f"🖼️→🎵 Processing: {image_path}")

        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            image = Image.open(BytesIO(img_path.read_bytes()))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot open image: {e}")

        img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.vgg16(img_tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        species_code = self.idx_to_class[pred_idx.item()]
        confidence   = float(conf.item())

        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)
        top3 = [
            {
                "species":    self.idx_to_class[idx.item()].replace("-", " ").replace("_", " "),
                "confidence": float(prob.item())
            }
            for prob, idx in zip(top3_probs[0], top3_idx[0])
        ]

        meta        = self.species_metadata.get(species_code, {})
        common_name = meta.get("common_name", species_code.replace("-", " ").replace("_", " "))
        scientific  = meta.get("scientific",  "")
        family      = meta.get("family",      "Aves")

        logger.info(f"🐦 Predicted: {common_name} | Confidence: {confidence:.2%}")

        output_filename = self._unique_filename("audio", img_path.stem, ".wav")
        output_path     = self.outputs_dir / output_filename
        audio_type      = self._generate_audio(species_code, output_path)

        return {
            "species":         common_name,
            "common_name":     common_name,
            "confidence":      confidence,
            "audio_file":      output_filename,
            "audio_type":      audio_type,
            "region":          "India",
            "scientific_name": scientific,
            "family":          family,
            "top3":            top3,
        }

    def audio_to_image(self, audio_path: str) -> dict:
        logger.info(f"🎵→🖼️  Processing: {audio_path}")

        aud_path = Path(audio_path)
        if not aud_path.exists():
            raise FileNotFoundError(f"Audio not found: {aud_path}")

        try:
            y, sr = librosa.load(str(aud_path), sr=self.audio_sr,
                                 duration=self.duration, mono=True)
            target_len = self.audio_sr * self.duration
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            mel    = librosa.feature.melspectrogram(
                y=y, sr=self.audio_sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_len
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            if mel_db.shape[1] < self.fixed_t:
                mel_db = np.pad(mel_db, ((0, 0), (0, self.fixed_t - mel_db.shape[1])))
            else:
                mel_db = mel_db[:, :self.fixed_t]

            mel_db     = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            mel_tensor = torch.tensor(mel_db, dtype=torch.float32)\
                             .unsqueeze(0).unsqueeze(0).to(self.device)

        except Exception as e:
            raise ValueError(f"Cannot process audio: {e}")

        with torch.no_grad():
            logits = self.audio_classifier(mel_tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        species_code = self.audio_idx_to_class[pred_idx.item()]
        confidence   = float(conf.item())

        meta        = self.species_metadata.get(species_code, {})
        common_name = meta.get("common_name", species_code.replace("-", " ").replace("_", " "))
        scientific  = meta.get("scientific",  "")
        family      = meta.get("family",      "Aves")

        logger.info(f"🐦 Predicted Audio: {common_name} | Confidence: {confidence:.2%}")

        output_filename = self._unique_filename("image", aud_path.stem, ".png")
        output_path     = self.outputs_dir / output_filename

        if self.image_gen:
            prompt = (
                f"high quality wildlife photograph of {common_name}, "
                f"Indian bird, natural habitat, sharp focus, detailed feathers"
            )
            try:
                img = self.image_gen(
                    prompt, num_inference_steps=25,
                    guidance_scale=7.0, height=512, width=512
                ).images[0]
                img.save(output_path)
                logger.info(f"  ✅ Stable Diffusion image saved")
            except Exception as e:
                logger.warning(f"⚠️  SD failed: {e}")
                self._generate_fallback_image(species_code, output_path)
        else:
            self._generate_fallback_image(species_code, output_path)

        return {
            "species":         common_name,
            "common_name":     common_name,
            "confidence":      confidence,
            "image_file":      output_filename,
            "region":          "India",
            "scientific_name": scientific,
            "family":          family,
        }


    # ──────────────────────────────────────────────────────────────────────────
    # AUDIO HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_audio(self, species_code: str, output_path: Path) -> str:
        """Returns 'real' or 'synthetic' depending on what was used."""
        logger.info(f"  🔍 Looking for audio: {species_code}")

        name_variants = [
            species_code,
            species_code.replace("-", "_"),
            species_code.replace("_", "-"),
            "-".join(w.capitalize() for w in species_code.replace("_", "-").split("-")),
        ]

        search_bases = [
            Path("data/raw/audio_dataset"),
            Path("data/raw/xeno_canto"),
        ]

        for base in search_bases:
            if not base.exists():
                continue
            for variant in name_variants:
                folder = base / variant
                if folder.exists():
                    for pattern in ["*.mp3", "*.wav", "*.ogg", "*.flac"]:
                        files = sorted(folder.glob(pattern))
                        if files:
                            try:
                                y, sr       = librosa.load(str(files[0]), sr=22050, duration=5.0)
                                y           = y / (np.max(np.abs(y)) + 1e-9)
                                audio_int16 = (y * 32767).astype(np.int16)
                                sf.write(str(output_path), audio_int16, 22050, subtype="PCM_16")
                                logger.info(f"  ✅ Real audio used: {files[0].name}")
                                return "real"
                            except Exception as e:
                                logger.error(f"  ❌ Real audio failed: {e}")

        # Fallback: synthetic chirp
        logger.info(f"  💡 Generating synthetic chirp for: {species_code}")
        seed       = self._stable_seed_from_species(species_code)
        sr         = 22050
        dur        = 4.0
        n          = int(sr * dur)
        base_freq  = 1500 + (seed % 3000)
        chirp_high = base_freq * 1.6
        num_chirps = 4 + (seed % 5)
        chirp_dur  = 0.1
        gap        = dur / num_chirps
        audio      = np.zeros(n)

        for i in range(num_chirps):
            start  = int(i * gap * sr)
            end    = min(start + int(chirp_dur * sr), n)
            length = end - start
            freq   = np.linspace(base_freq, chirp_high, length)
            phase  = 2 * np.pi * np.cumsum(freq) / sr
            env    = np.hanning(length)
            audio[start:end] += np.sin(phase) * env
            audio[start:end] += 0.3 * np.sin(2 * phase) * env

        audio       = audio / (np.max(np.abs(audio)) + 1e-9)
        audio_int16 = (audio * 32767).astype(np.int16)
        sf.write(str(output_path), audio_int16, sr, subtype="PCM_16")
        logger.info(f"  ✅ Synthetic audio saved: {output_path.name}")
        return "synthetic"


    # ──────────────────────────────────────────────────────────────────────────
    # IMAGE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_fallback_image(self, species_code: str, output_path: Path):
        """Use real iBC53 dataset image if available, else styled placeholder."""

        name_variants = [
            species_code,
            species_code.replace("_", "-"),
            species_code.replace(" ", "-"),
            "-".join(w.capitalize() for w in species_code.replace("_", "-").split("-")),
            "-".join(w.capitalize() for w in species_code.replace(" ", "-").split("-")),
        ]

        search_bases = [
            Path("data/raw/indian-birds/Birds_25/train"),
            Path("data/raw/indian-birds/Birds_25/valid"),
            Path("data/raw/ibc53/train"),
            Path("data/raw/ibc53/val"),
            Path("data/raw/ibc53"),
        ]

        for base in search_bases:
            if not base.exists():
                continue
            for variant in name_variants:
                folder = base / variant
                if folder.exists():
                    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                        files = sorted(folder.glob(pattern))
                        if not files:
                            continue

                        # Try up to 5 candidates, pick highest detail score
                        candidates = files[:5]
                        best_img   = None
                        best_score = -1.0

                        for f in candidates:
                            try:
                                candidate = Image.open(f)
                                candidate = ImageOps.exif_transpose(candidate)  # fix rotation
                                candidate = candidate.convert("RGB")

                                # Score by pixel std — higher = more detail = better shot
                                arr   = np.array(candidate.resize((64, 64)))
                                score = float(arr.std())

                                logger.info(f"  📊 Candidate {f.name}: score={score:.1f}")

                                if score > best_score:
                                    best_score = score
                                    best_img   = candidate

                            except Exception as e:
                                logger.warning(f"  ⚠️  Skipping {f.name}: {e}")

                        if best_img:
                            best_img = best_img.resize((512, 512), Image.LANCZOS)
                            best_img.save(output_path)
                            logger.info(f"  ✅ Best image saved (score={best_score:.1f}): {output_path.name}")
                            return

        logger.warning(f"  ⚠️  No dataset image found for '{species_code}', using placeholder")
        self._draw_styled_placeholder(species_code, output_path)

    def _draw_styled_placeholder(self, species_code: str, output_path: Path):
        seed  = self._stable_seed_from_species(species_code)
        r_off = seed % 12
        g_off = (seed >> 4) % 18
        bg    = (8 + r_off, 22 + g_off, 12 + r_off)
        img   = Image.new("RGB", (512, 512), color=bg)
        draw  = ImageDraw.Draw(img)

        # Subtle grid
        for i in range(0, 512, 32):
            draw.line([(i, 0), (i, 512)], fill=(bg[0]+4, bg[1]+4, bg[2]+4), width=1)
            draw.line([(0, i), (512, i)], fill=(bg[0]+4, bg[1]+4, bg[2]+4), width=1)

        # Glow rings behind bird
        for r in range(90, 30, -15):
            alpha = int(15 * (90 - r) / 60)
            draw.ellipse(
                [256-r, 190-r, 256+r, 190+r],
                fill=(40+alpha, 110+alpha, 65+alpha)
            )

        # Bird silhouette
        cx, cy = 256, 195
        draw.ellipse([cx-42, cy-26, cx+42, cy+26], fill=(55, 140, 80))
        draw.ellipse([cx+22, cy-38, cx+58, cy+4],  fill=(55, 140, 80))
        draw.ellipse([cx+32, cy-28, cx+44, cy-14], fill=(bg[0]+5, bg[1]+5, bg[2]+5))
        draw.polygon([(cx-42, cy+5), (cx-80, cy+28), (cx-28, cy+14)], fill=(45, 118, 65))
        draw.polygon([(cx+22, cy+18), (cx+10, cy+46), (cx+36, cy+46)], fill=(45, 118, 65))

        # Text panel
        common = species_code.replace("-", " ").replace("_", " ").title()
        draw.rectangle([0, 395, 512, 512], fill=(4, 14, 7))
        draw.line([(0, 395), (512, 395)], fill=(55, 140, 80), width=1)
        name_x = max(20, 256 - len(common) * 5)
        draw.text((name_x, 410), common,                        fill=(109, 202, 138))
        draw.text((148,    440), "BioSync · iBC53 Dataset",     fill=(60, 120, 78))
        draw.text((122,    465), "Stable Diffusion not loaded",  fill=(38, 78, 52))
        draw.text((175,    488), "Species identified ✓",         fill=(55, 140, 80))

        img.save(output_path)
        logger.info(f"  ✅ Styled placeholder saved: {output_path.name}")


    # ──────────────────────────────────────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────────────────────────────────────

    def _unique_filename(self, prefix: str, stem: str, ext: str) -> str:
        uid       = uuid.uuid4().hex[:8]
        safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
        return f"{prefix}_{safe_stem}_{uid}{ext}"

    def _stable_seed_from_species(self, species: str) -> int:
        h = hashlib.sha256(species.encode("utf-8")).hexdigest()
        return int(h[:8], 16)