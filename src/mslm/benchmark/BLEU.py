import os
import sacrebleu
import torch
import gc
from tqdm import tqdm
import torch.nn.functional as F
import h5py
import numpy as np
from ..dataloader import KeypointDataset, collate_fn
from torch.utils.data import DataLoader
from ..utils.setup_train import prepare_datasets, build_model, setup_paths, BatchSampler
from ..utils.config_loader import cfg
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sacrebleu

device = "cuda" if torch.cuda.is_available() else "cpu"

HYPS = [
  ["a tierra", "a tierra"],
  ["abecedario", "abecedario.mp4"],
  ["abrir", "abrir"],
  ["abrir cortina", "abrir-cortina"],
  ["aburrido", "aburrido"],
  ["aceptar", "aceptar"],
  ["aceptaron", "aceptaron"],
  ["acercarse", "acercarse"],
  ["acuerdos", "acuerdos"],
  ["adivina", "adivina"],
  ["agua", "agua"],
  ["ah!", "aaah!"],
  ["ahi", "ahi", "ahí"],
  ["ahora", "ahora"],
  ["aire viento", "aire/ viento"],
  ["algunos", "algunos"],
  ["alla", "alla", "allá"],
  ["alo", "aló"],
  ["alto", "alto"],
  ["amanecer", "amanecer"],
  ["amargo", "amargo"],
  ["amarillo", "amarillo"],
  ["ambos", "ambos"],
  ["ambulancia", "ambulancia"],
  ["amigo", "amigo"],
  ["anillo", "anillo"],
  ["anochecer", "anochecer"],
  ["antes", "antes"],
  ["antigua", "antigua"],
  ["apaga", "apaga"],
  ["apellido", "apellido"],
  ["aprender", "aprender"],
  ["argentina", "argentina"],
  ["arrodillarse postrarse", "arrodillarse/ postrarse"],
  ["arroz", "arroz"],
  ["atardecer", "atardecer"],
  ["atrapar", "atrapar"],
  ["averiguar", "averiguar"],
  ["avion", "avion"],
  ["avioneta", "avioneta"],
  ["ayuda", "ayuda"],
  ["ayudar", "ayudar"],
  ["azul claro", "azul claro"],
  ["bailar", "bailar"],
  ["bajar escalera", "bajar-escalera"],
  ["banar", "bañar"],
  ["barbacoa", "barbacoa"],
  ["barco", "barco"],
  ["bastante", "bastante"],
  ["bicicleta", "bicicleta"],
  ["bien", "bien"],
  ["bola", "bola"],
  ["bola de cristal", "bola-de-cristal"],
  ["bote", "bote"],
  ["brillante", "brillante"],
  ["burlarse", "burlarse"],
  ["bus", "bus"],
  ["busca", "busca"],
  ["buscar", "buscar"],
  ["c i l m a", "c-i-l-m-a"],
  ["c l", "c-l"],
  ["caer", "caer"],
  ["caja", "caja"],
  ["cajon", "cajón"],
  ["callada", "callada"],
  ["calor", "calor"],
  ["cama", "cama"],
  ["cambiarse de ropa", "cambiarse de ropa"],
  ["caminar", "caminar"],
  ["camion", "camion"],
  ["camioneta", "camioneta"],
  ["campamento", "campamento"],
  ["campamento carpa iman", "campamento/ carpa/ imán "],
  ["captar", "captar"],
  ["cara", "cara"],
  ["caramelo", "caramelo"],
  ["cargar", "cargar"],
  ["carro", "carro"],
  ["cartas de tarot", "cartas-de-tarot"],
  ["casa", "casa"],
  ["casaca", "casaca"],
  ["casar", "casar"],
  ["catorce", "catorce"],
  ["cerrar", "cerrar"],
  ["cerrar cajon", "cerrar-cajon"],
  ["cerrar cortina", "cerrar cortina"],
  ["chau", "chau"],
  ["chocar", "chocar"],
  ["cien", "cien"],
  ["cincuenta", "cincuenta"],
  ["ciudad", "ciudad"],
  ["colores", "colores"],
  ["combi", "combi"],
  ["comer", "comer"],
  ["comprar", "comprar"],
  ["contactar", "contactar"],
  ["contar", "contar"],
  ["contar dinero", "contar-dinero"],
  ["contar numeros", "contar-numeros"],
  ["convive o se junto", "convive ó se juntó"],
  ["conyuge", "conyuge"],
  ["copiar", "copiar"],
  ["cornudo", "cornudo"],
  ["correr", "correr"],
  ["cortar", "cortar "],
  ["cortar interrumpida", "cortar/ interrumpida"],
  ["cortina", "cortina"],
  ["cortina abierta", "cortina abierta"],
  ["cortinas", "cortinas"],
  ["cual", "cual"],
  ["cuando", "cuando"],
  ["cuanto", "cuanto"],
  ["cuarenta", "cuarenta"],
  ["cuarto", "cuarto"],
  ["cuatro", "cuatro"],
  ["cuerpo", "cuerpo"],
  ["cumpleanos", "cumpleaños"],
  ["cuna", "cuna"],
  ["dar", "dar"],
  ["dar pasos", "dar-pasos"],
  ["darse cuenta de", "darse cuenta de"],
  ["debajo", "debajo"],
  ["decir", "decir"],
  ["decirme", "decirme"],
  ["dentro", "dentro"],
  ["desaparecer", "desaparecer"],
  ["desaparecido", "desaparecido"],
  ["desayuno", "desayuno"],
  ["despues", "despues"],
  ["despues o siguiente", "después ó siguiente"],
  ["detalles caracteristicas perfiles", "detalles/ caracteristicas/ pérfiles"],
  ["dia", "dia"],
  ["dibujo", "dibujo"],
  ["dice", "dice"],
  ["diecinueve", "diecinueve"],
  ["dieciocho", "dieciocho"],
  ["dieciseis", "dieciseis"],
  ["diecisiete", "diecisiete"],
  ["diez", "diez"],
  ["dificil", "dificil", "difícil"],
  ["dijo", "dijo"],
  ["dinero", "dinero"],
  ["doce", "doce"],
  ["donde", "donde", "dónde"],
  ["dormir", "dormir"],
  ["dos", "dos"],
  ["dos se acercan", "dos se acercan"],
  ["dueno propiedad de alguien", "dueño(a)/ propiedad de alguien"],
  ["ejercicio", "ejercicio"],
  ["el", "el", "él"],
  ["el fue", "el fue"],
  ["el otro viene", "el otro viene"],
  ["ella", "ella"],
  ["ellos", "ellos"],
  ["empezar", "empezar"],
  ["encontrar", "encontrar"],
  ["encontrar acercar", "encontrar/ acercar"],
  ["encontrar o acercar", "encontrar o acercar"],
  ["encontrarse o acercarse", "encontrarse o acercarse"],
  ["enemigo", "enemigo"],
  ["enganar", "engañar"],
  ["engordar", "engordar"],
  ["entender", "entender"],
  ["entendiste?", "entendiste?"],
  ["entrar", "entrar"],
  ["entrar adentro", "entrar/ adentro"],
  ["esa ella", "esa/ ella"],
  ["esa mujer", "esa mujer"],
  ["escapar fugar", "escapar/ fugar"],
  ["esconder", "esconder"],
  ["esconderse", "esconderse"],
  ["escribir", "escribir"],
  ["escuchar", "escuchar"],
  ["ese hombre", "ese hombre"],
  ["ese? yo?", "ese? yo?"],
  ["esos", "esos", "ese", "ese"],
  ["espaguetis", "espaguetis"],
  ["espejo", "espejo"],
  ["esperar", "esperar"],
  ["espumadera", "espumadera"],
  ["estar bien", "estar-bien"],
  ["este", "este", "este(a)"],
  ["este esta", "este/ esta"],
  ["este esta ella", "este/esta/ella"],
  ["falta", "falta"],
  ["faltar", "faltar"],
  ["familia", "familia"],
  ["feliz", "feliz"],
  ["fin", "fin"],
  ["flaco", "flaco"],
  ["fortachon", "fortachon"],
  ["foto", "foto"],
  ["fregado", "fregado"],
  ["frio", "frio", "frío"],
  ["fue", "fue"],
  ["fundar"],
  ["futuro", "futuro"],
  ["g j o n", "g-j-o-n"],
  ["goma de mascar"],
  ["gordo"],
  ["gorro"],
  ["gracias"],
  ["grande"],
  ["guardar"],
  ["hablamos"],
  ["hablar"],
  ["hacer"],
  ["hacer preguntas", "hacer preguntas"],
  ["hambriento"],
  ["helicoptero"],
  ["hijo", "hijo", "hijo(a)"],
  ["historia"],
  ["hola"],
  ["hombre"],
  ["hoy"],
  ["hum?", "hum?", "hummm?", "hummm???", "hummm? sí"],
  ["idea"],
  ["igual" ],
  ["imposible"],
  ["infiel"],
  ["ir", "irse"],
  ["ix oculto", "ix-oculto"],
  ["j o n", "j-o-n"],
  ["j o n y s u e", "j-o-n-y-s-u-e"],
  ["j u a n", "j-u-a-n"],
  ["joven"],
  ["jovenes", "jovenes", "jovenes (chicas)", "jovenes chicos", "jovenes/chicos"],
  ["jugar"],
  ["juntos"],
  ["juntos en grupo", "juntos/ en grupo"],
  ["lancha",],
  ["leche"],
  ["leche dulce", "leche dulce"],
  ["lejos", "lejos"],
  ["lentes", "lentes"],
  ["lentes de sol", "lentes-de-sol", "lentes de sol oracion", "lentes-de-sol-oracion"],
  ["linterna", "linternas"],
  ["llamada", "llamar"],
  ["llegar", "llegar"],
  ["lo siento", "lo-siento"],
  ["lo tres", "lo tres", "los tres"],
  ["luego", "luego"],
  ["malo", "malo"],
  ["mandar", "mandar"],
  ["mapa", "mapa"],
  ["matrimonio", "matrimonio"],
  ["matrimonio boda", "matrimonio/ boda"],
  ["me dicen", "me dicen"],
  ["mejor", "mejor"],
  ["mi", "mi"],
  ["mochila", "mochila"],
  ["moneda", "moneda"],
  ["mucho", "mucho"],
  ["mucho dinero", "mucho-dinero"],
  ["mujer", "mujer", "mujeres", "mujeres"],
  ["musica", "música"],
  ["nacer", "nacer"],
  ["nada mas fin", "nada más/ fin"],
  ["navio", "navío"],
  ["ninguno", "ninguno"],
  ["nino", "niño"],
  ["no", "no", "no ", "no- no- no", "no-no", "no-no-no"],
  ["no importa", "no importa "],
  ["no sabe"],
  ["nombre"],
  ["nosotros", "nosotros"],
  ["nube", "nube", "nube clima", "nube/ clima"],
  ["o", "ó"],
  ["objetivo proposito", "objetivo/ proposito"],
  ["ok", "ok"],
  ["opaco", "opaco"],
  ["oscuro", "oscuro"],
  ["otro", "otro", "otro(a)", "otro uno", "otro uno"],
  ["oye", "oye"],
  ["p d", "p-d"],
  ["p e d r o", "p-e-d-r-o"],
  ["paciencia", "paciencia"],
  ["pais", "país"],
  ["panzon con rollos", "panzón / con rollos"],
  ["parada", "parada"],
  ["paragua", "paragua"],
  ["parecer", "parecer"],
  ["pelota", "pelota"],
  ["pensando o razonando", "pensar", "pensar razonar", "pensar/ razonar", "pensar razonar reflexionar", "pensar/ razonar/ reflexionar"],
  ["pequeno", "pequeño"],
  ["pequeno chico", "pequeño/ chico"],
  ["perder", "perder"],
  ["perfume", "perfume"],
  ["perseguir seguir", "perseguir/ seguir"],
  ["persona baja del bote", "persona baja del bote"],
  ["persona pequena", "persona-pequeña"],
  ["pescar", "pescar"],
  ["pez", "pez"],
  ["pila", "pila"],
  ["policia", "policia", "policía"],
  ["preguntar", "preguntar"],
  ["presente", "presente"],
  ["primero", "primero", "primero "],
  ["probar", "probar"],
  ["pronto en breve", "pronto/ en breve"],
  ["proposito objetivo", "próposito/ objetivo"],
  ["prueba", "prueba"],
  ["puerta", "puerta"],
  ["que", "que?", "qué"],
  ["que hace?", "qué hace?"],
  ["querer", "querer"],
  ["quien", "quien"],
  ["quinto", "quinto"],
  ["r a f a e l", "r-a-f-a-e-l"],
  ["razonando", "razonando"],
  ["rojo", "rojo"],
  ["ropa", "ropa"],
  ["rosado", "rosado"],
  ["s u e", "s-u-e"],
  ["saber", "saber"],
  ["salir vamos", "salir/ vamos"],
  ["saludan", "saludan"],
  ["se fue", "se fue"],
  ["se fue salio", "se fue/ salió", "se fue salir", "se fue/ salir"],
  ["se separa", "se separa"],
  ["segundo", "segundo"],
  ["sentir", "sentir"],
  ["sexto", "sexto"],
  ["si", "sí", "sí - sí - sí"],
  ["sol", "sol"],
  ["sordo", "sordo"],
  ["sorprendida se asombra", "sorprendida/ se asombra"],
  ["subir persona", "subir-persona"],
  ["sudar", "sudar"],
  ["suludos", "suludos"],
  ["telefono", "telefono"],
  ["tercero", "tercero"],
  ["termine", "terminé"],
  ["timida", "tímida"],
  ["timida con roche", "tímida/ con roche"],
  ["titulo", "titulo"],
  ["todo juntos", "todo juntos"],
  ["trampa", "trampa"],
  ["tranquila", "tranquila"],
  ["tres", "tres"],
  ["trotar correr", "trotar/ correr"],
  ["tu", "tú"],
  ["tu o el", "tú ó él"],

  ["un una", "un/ una"],
  ["uno", "uno", "un uno", "un? uno?", "un/ uno", "un/ uno(a)"],

  ["uruguay", "uruguay"],
  ["varios", "varios"],
  ["ver", "ver"],
  ["verde", "verde"],
  ["verguenza", "verguenza"],
  ["vio miro", "vió/ miró"],
  ["vio o miro", "vió ó miró"],
  ["viveres", "víveres"],
  ["ya", "ya"],
  ["ya se", "ya sé"],
  ["yo", "yo"],
  ["yogur", "yogur"],
]

def bleu_collate_fn(batch):
    keypoints, mask_data, _, mask_embds, labels = collate_fn(batch)
    return keypoints, mask_data, mask_embds, labels

def exec_bleu(true:list[list], pred:list[str]):
    """Compute the BLEU score for the given true and predicted sentences.

    Args:
        true (list[list]): lista de listas de oraciones verdaderas, cada lista contienen varias variantes de la misma oración.
        pred (list[str]): lista de oraciones generadas por el modelo.

    Returns:
        float: el puntaje BLEU calculado.
    """
    bleu = sacrebleu.corpus_bleu(pred, true, smooth_method="exp")
    return bleu.score

def load_config():
    _,_, h5_file = setup_paths()
        
    training_cfg:dict = cfg.training
    model_cfg = cfg.model
    model_cfg["device"] = "cuda"
    return h5_file, training_cfg, model_cfg

def load_dataset(h5_file, key_points:int):
    keypoint_reader = KeypointDataset(h5Path=h5_file, return_label=True, n_keypoints=key_points, data_augmentation=False, max_length=4000)
    train_dataset, _, _, _ = keypoint_reader.split_dataset(1)
    train_sampler = BatchSampler(train_dataset, 1)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=bleu_collate_fn,
        batch_sampler=train_sampler
    )
    return train_dataloader, keypoint_reader.id_to_label, keypoint_reader.label_to_id

def load_model(model_parameters:dict, version:str, checkpoint:str, epoch:int):
    model_parameters.pop("device", None)  # remove device from model parameters
    print(model_parameters)
    model = build_model(**model_parameters)    

    model_location = f"../outputs/checkpoints/{version}/{checkpoint}/{epoch}/checkpoint.pth" 
    if not os.path.exists(model_location):
        raise FileNotFoundError(
            f"Model not found {model_location}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_location, map_location=device)

    model.load_state_dict(state_dict["model_state"])
    model.to(device)

    return model

def load_llm(model_id="unsloth/Llama-3.2-3B"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # o "fp4"
        bnb_4bit_compute_dtype=torch.bfloat16,  # o torch.float16 si no tienes soporte bf16
    )

    llama_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embeddings = llama_model.get_input_embeddings().weight.data
    del llama_model
    return tokenizer, embeddings

def get_idx_hyps(word):
    for i, cluster in enumerate(HYPS):
        if word in cluster:
            return i
    return -1

def embeddings_to_text(embeddings: torch.Tensor, all_embeddings: torch.Tensor, tokenizer) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_matrix = all_embeddings.to(device)  # [V, D]

    target_dtype = embedding_matrix.dtype
    embeddings = embeddings.to(device=device, dtype=target_dtype)
    #print("Embeddings device:", embeddings.device, "| Matrix device:", embedding_matrix.device)

    embedding_matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)  # [V, D]
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [T, D]
    similarities = torch.matmul(embeddings_norm, embedding_matrix_norm.T)  # [T, V]
    token_ids = torch.argmax(similarities, dim=1).tolist()
    #print(f"Token IDs: {token_ids}")
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _is_space_token(tok: str) -> bool:
    # En tokenizers tipo SentencePiece, '▁' marca inicio de palabra
    return tok.startswith("▁") or tok.startswith("<bos>")

@torch.no_grad()
def embeddings_to_text_viterbi(
    embeddings: torch.Tensor,        # [L, D] (viene de tu modelo o del HDF5)
    all_embeddings: torch.Tensor,    # [V, D] (tabla del LLM)
    tokenizer,
    topk: int = 24,
    tau: float = 0.30,
    rep_penalty: float = 0.65,
    nospace_run_penalty: float = 0.10,
    start_space_bonus: float = 0.35,
):
    # Unificar device y dtype
    device = all_embeddings.device
    dtype  = all_embeddings.dtype

    # eps evita NaNs si llega un vector ~cero
    X = F.normalize(embeddings.to(device=device, dtype=dtype), p=2, dim=1, eps=1e-6)  # [L, D]
    E = F.normalize(all_embeddings.to(device=device, dtype=dtype), p=2, dim=1, eps=1e-6)  # [V, D]

    S = torch.matmul(X, E.T)  # [L, V]  <-- ya no crashea
    topv, topi = torch.topk(S, k=min(topk, S.size(1)), dim=1)

    special = set(tokenizer.all_special_ids)
    tok_str = tokenizer.convert_ids_to_tokens(torch.arange(E.size(0), device=device).tolist())

    # DP/Viterbi (igual que te pasé antes) ...
    # --- inicialización ---
    dp   = torch.full((topv.size(0), topv.size(1)), -1e9, device=device)
    prev = torch.full_like(dp, -1, dtype=torch.long)

    def is_space(tok: str) -> bool:
        return tok.startswith("▁")

    # t=0
    for j in range(topv.size(1)):
        vid = int(topi[0, j].item())
        if vid in special: 
            continue
        sc = float(topv[0, j].item())
        if sc < tau:
            continue
        if is_space(tok_str[vid]): 
            sc += start_space_bonus
        dp[0, j] = sc

    # transiciones
    for t in range(1, topv.size(0)):
        for j in range(topv.size(1)):
            vj = int(topi[t, j].item())
            if vj in special:
                continue
            base = float(topv[t, j].item())
            if base < tau:
                continue
            cur_space = is_space(tok_str[vj])

            best_val = -1e9
            best_k   = -1
            for k in range(topv.size(1)):
                prev_val = float(dp[t-1, k].item())
                if prev_val <= -1e8:
                    continue
                vi = int(topi[t-1, k].item())
                val = prev_val + base
                if vi == vj:
                    val -= rep_penalty
                prev_space = is_space(tok_str[vi])
                if not prev_space and not cur_space:
                    val -= nospace_run_penalty
                if val > best_val:
                    best_val = val
                    best_k   = k
            dp[t, j]   = best_val
            prev[t, j] = best_k

    # backtrack
    last_t = topv.size(0) - 1
    j = int(torch.argmax(dp[last_t]).item())
    if dp[last_t, j].item() <= -1e8:
        return ""

    ids = []
    for t in range(last_t, -1, -1):
        ids.append(int(topi[t, j].item()))
        j = int(prev[t, j].item())
        if t > 0 and j < 0:
            break
    ids.reverse()

    # limpieza
    ids2 = []
    for vid in ids:
        if vid in special:
            continue
        if not ids2 or ids2[-1] != vid:
            ids2.append(vid)

    pieces = tokenizer.convert_ids_to_tokens(ids2)
    text = ''.join(p.replace('▁', ' ') for p in pieces).strip()
    return text

def main(version:str, checkpoint:str, epoch:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_file, training_cfg, model_cfg = load_config()
    dataset, id_to_label, label_to_id = load_dataset(h5_file, model_cfg.get("n_keypoints", 111))
     
    model = load_model(model_cfg, version, checkpoint, epoch)
    model = model.to(device)
    model.eval()
    
    cache_path = "bleu_imitator_embeds.h5"
    dataset_size = len(dataset)
    results = []
    use_cached_results = False

    if os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as h5f:
            samples_group = h5f.get("samples")
            stored_size = h5f.attrs.get("dataset_size")
            processed_size = h5f.attrs.get("processed_samples")
            if (
                samples_group is not None
                and len(samples_group) == dataset_size
                and stored_size == dataset_size
                and processed_size == dataset_size
            ):
                for key in sorted(samples_group.keys()):
                    sample_group = samples_group[key]
                    label_value = sample_group.attrs.get("label", "")
                    if isinstance(label_value, bytes):
                        label_value = label_value.decode("utf-8")
                    hyps_data = sample_group["hyps"][()]
                    hyps_list = hyps_data.tolist() if isinstance(hyps_data, np.ndarray) else list(hyps_data)
                    hyps_list = [item.decode("utf-8") if isinstance(item, bytes) else item for item in hyps_list]
                    results.append({
                        "label": label_value,
                        "embed_pred": sample_group["embed_pred"][()],
                        "hyps": hyps_list
                    })
                use_cached_results = True
                print("Loaded cached BLEU embeddings.")
    
    if not use_cached_results:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(cache_path, "w") as h5f:
            h5f.attrs["dataset_size"] = dataset_size
            samples_group = h5f.create_group("samples")

            for sample_idx, (keypoints, mask_data, mask_embds, label_id) in enumerate(tqdm(dataset, desc="Processing samples")):
                label_text = id_to_label[label_id[0]]
                idx = get_idx_hyps(label_text)
                if idx == -1:
                    print("no hay mapeado}", label_text)
                    continue
                
                with torch.inference_mode():
                    data = keypoints.to(device=device, dtype=torch.float32, non_blocking=True)
                    mask_data = mask_data.to(device, non_blocking=True)

                    sign_embed, pool_embed = model(data, mask_data)
                    del pool_embed
                    
                    pred_embeds = sign_embed.to(device=device, dtype=torch.float32)
                    valid_tokens = (~mask_embds.to(pred_embeds.device, non_blocking=True)[0]).sum().item()
                    seq_len = min(pred_embeds.size(1), valid_tokens)
                    pred_embeds = pred_embeds[0, :seq_len, :].contiguous()

                    embed_array = pred_embeds.detach().cpu().numpy().astype("float32")
                    
                    res = {
                        "label": label_text,
                        "embed_pred": embed_array,
                        "hyps": HYPS[idx]
                    }
                    results.append(res)
                    
                    sample_group = samples_group.create_group(f"{sample_idx:06d}")
                    sample_group.attrs["label"] = label_text
                    sample_group.create_dataset("embed_pred", data=embed_array, compression="gzip")

                    hyps_array = np.array(res["hyps"], dtype=object)
                    sample_group.create_dataset("hyps", data=hyps_array, dtype=string_dtype)
                    
                    del data, mask_data, sign_embed, pred_embeds
                    gc.collect()
                    torch.cuda.empty_cache()

            h5f.attrs["processed_samples"] = len(samples_group)
    
    tokenizer, all_embeddings = load_llm()
    hyps_list_bench = []
    pred_list_bench = []
    for res in results:
        embed_pred = torch.from_numpy(res["embed_pred"])
        embed_pred = embed_pred.to(device=all_embeddings.device, dtype=all_embeddings.dtype)

        pred_text = embeddings_to_text_viterbi(embed_pred, all_embeddings, tokenizer)
        print(pred_text)
        
        pred_list_bench.append(pred_text)
        hyps_list_bench.append(res["hyps"])
    
    score_bleu = exec_bleu(hyps_list_bench, pred_list_bench)
    print(f"BLEU score: {score_bleu:.2f}")
    return score_bleu
