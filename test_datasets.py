import urllib.request
import json
import ssl

def check_hf_repo(repo_id):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ BAŞARILI: {repo_id}")
                return True
    except Exception as e:
        print(f"❌ BAŞARISIZ: {repo_id} ({type(e).__name__})")
        return False

repos = [
    "merve/turkish_instructions",
    "TFLai/Turkish-Alpaca",
    "malhajar/alpaca-gpt4-tr",
    "ucekmez/OpenOrca-tr"
]

print("🔍 HF Hub Dataset Kontrolü Başlıyor...\n")
for r in repos:
    check_hf_repo(r)
