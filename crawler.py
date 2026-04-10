import os
import requests
import time

PAGES_DIR = "pages"
INDEX_FILE = "index.txt"

# 100+ English Wikipedia articles (text-heavy, same language)
URLS = [
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Java_(programming_language)",
    "https://en.wikipedia.org/wiki/C_(programming_language)",
    "https://en.wikipedia.org/wiki/JavaScript",
    "https://en.wikipedia.org/wiki/HTML",
    "https://en.wikipedia.org/wiki/CSS",
    "https://en.wikipedia.org/wiki/World_Wide_Web",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Robotics",
    "https://en.wikipedia.org/wiki/Data_science",
    "https://en.wikipedia.org/wiki/Big_data",
    "https://en.wikipedia.org/wiki/Cloud_computing",
    "https://en.wikipedia.org/wiki/Blockchain",
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://en.wikipedia.org/wiki/Bitcoin",
    "https://en.wikipedia.org/wiki/Ethereum",
    "https://en.wikipedia.org/wiki/Operating_system",
    "https://en.wikipedia.org/wiki/Linux",
    "https://en.wikipedia.org/wiki/Microsoft_Windows",
    "https://en.wikipedia.org/wiki/MacOS",
    "https://en.wikipedia.org/wiki/Android_(operating_system)",
    "https://en.wikipedia.org/wiki/IOS",
    "https://en.wikipedia.org/wiki/Database",
    "https://en.wikipedia.org/wiki/SQL",
    "https://en.wikipedia.org/wiki/NoSQL",
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Data_structure",
    "https://en.wikipedia.org/wiki/Graph_theory",
    "https://en.wikipedia.org/wiki/Sorting_algorithm",
    "https://en.wikipedia.org/wiki/Search_algorithm",
    "https://en.wikipedia.org/wiki/Encryption",
    "https://en.wikipedia.org/wiki/Cybersecurity",
    "https://en.wikipedia.org/wiki/Firewall_(computing)",
    "https://en.wikipedia.org/wiki/Software_engineering",
    "https://en.wikipedia.org/wiki/Agile_software_development",
    "https://en.wikipedia.org/wiki/DevOps",
    "https://en.wikipedia.org/wiki/Version_control",
    "https://en.wikipedia.org/wiki/Git",
    "https://en.wikipedia.org/wiki/Open-source_software",
    "https://en.wikipedia.org/wiki/Linux_kernel",
    "https://en.wikipedia.org/wiki/TCP/IP",
    "https://en.wikipedia.org/wiki/HTTP",
    "https://en.wikipedia.org/wiki/Domain_Name_System",
    "https://en.wikipedia.org/wiki/Email",
    "https://en.wikipedia.org/wiki/Search_engine",
    "https://en.wikipedia.org/wiki/Google_Search",
    "https://en.wikipedia.org/wiki/Web_crawler",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/PageRank",
    "https://en.wikipedia.org/wiki/TF%E2%80%93IDF",
    "https://en.wikipedia.org/wiki/Inverted_index",
    "https://en.wikipedia.org/wiki/Boolean_algebra",
    "https://en.wikipedia.org/wiki/Turing_machine",
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/John_von_Neumann",
    "https://en.wikipedia.org/wiki/Ada_Lovelace",
    "https://en.wikipedia.org/wiki/Charles_Babbage",
    "https://en.wikipedia.org/wiki/Tim_Berners-Lee",
    "https://en.wikipedia.org/wiki/Linus_Torvalds",
    "https://en.wikipedia.org/wiki/Dennis_Ritchie",
    "https://en.wikipedia.org/wiki/Guido_van_Rossum",
    "https://en.wikipedia.org/wiki/Elon_Musk",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Tesla,_Inc.",
    "https://en.wikipedia.org/wiki/Apple_Inc.",
    "https://en.wikipedia.org/wiki/Google",
    "https://en.wikipedia.org/wiki/Microsoft",
    "https://en.wikipedia.org/wiki/Amazon_(company)",
    "https://en.wikipedia.org/wiki/Facebook",
    "https://en.wikipedia.org/wiki/Twitter",
    "https://en.wikipedia.org/wiki/Netflix",
    "https://en.wikipedia.org/wiki/Wikipedia",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Virtual_reality",
    "https://en.wikipedia.org/wiki/Augmented_reality",
    "https://en.wikipedia.org/wiki/3D_printing",
    "https://en.wikipedia.org/wiki/Nanotechnology",
    "https://en.wikipedia.org/wiki/Biotechnology",
    "https://en.wikipedia.org/wiki/Genetic_engineering",
    "https://en.wikipedia.org/wiki/CRISPR_gene_editing",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Renewable_energy",
    "https://en.wikipedia.org/wiki/Solar_energy",
    "https://en.wikipedia.org/wiki/Wind_power",
    "https://en.wikipedia.org/wiki/Nuclear_power",
    "https://en.wikipedia.org/wiki/Electric_vehicle",
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/Mars",
    "https://en.wikipedia.org/wiki/Moon",
    "https://en.wikipedia.org/wiki/Solar_System",
    "https://en.wikipedia.org/wiki/Milky_Way",
    "https://en.wikipedia.org/wiki/Black_hole",
    "https://en.wikipedia.org/wiki/Theory_of_relativity",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Calculus",
    "https://en.wikipedia.org/wiki/Linear_algebra",
    "https://en.wikipedia.org/wiki/Statistics",
    "https://en.wikipedia.org/wiki/Probability_theory",
    "https://en.wikipedia.org/wiki/Game_theory",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentCrawler/1.0"
}


def crawl():
    os.makedirs(PAGES_DIR, exist_ok=True)

    index_lines = []
    success_count = 0

    for i, url in enumerate(URLS, start=1):
        print(f"[{i}/{len(URLS)}] Downloading: {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding

            filename = f"{i}.txt"
            filepath = os.path.join(PAGES_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(resp.text)

            index_lines.append(f"{i}\t{url}")
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(0.5)  # polite delay between requests

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    print(f"\nDone! Downloaded {success_count}/{len(URLS)} pages.")
    print(f"Pages saved to: {PAGES_DIR}/")
    print(f"Index saved to: {INDEX_FILE}")


if __name__ == "__main__":
    crawl()
