"""Script untuk melakukan push repository ke GitHub via HTTPS."""

import getpass
import os
import sys
from dulwich import porcelain


def main():
    print("=" * 60)
    print("PUSH REPOSITORY KE GITHUB (HTTPS)")
    print("=" * 60)
    print(
        "Target: https://github.com/MuhammaDZhainalFirdaus/MuhammadZhainalFirdaus-301220041-Skripsi.git"
    )
    print()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        if len(sys.argv) > 1:
            token = sys.argv[1].strip()
        else:
            print("Untuk push lewat HTTPS, GitHub memerlukan Personal Access Token (PAT).")
            print("Buat token di: https://github.com/settings/tokens/new")
            print("(Beri centang pada izin 'repo', lalu klik 'Generate token')\n")
            token = getpass.getpass(
                "Masukkan GitHub Personal Access Token (teks tidak akan tampil): "
            ).strip()

    if not token:
        print("\n[BATAL] Token tidak boleh kosong.")
        sys.exit(1)

    username = "MuhammaDZhainalFirdaus"
    push_url = (
        f"https://{username}:{token}@github.com/"
        f"MuhammaDZhainalFirdaus/MuhammadZhainalFirdaus-301220041-Skripsi.git"
    )

    print("\nSedang mengunggah (push) ke branch main...")
    repo = porcelain.open_repo(".")
    try:
        porcelain.push(repo, push_url, "refs/heads/main")
        print("\n[BERHASIL] Repository Anda telah sukses di-push ke GitHub!")
        print(
            "Lihat di: https://github.com/MuhammaDZhainalFirdaus/MuhammadZhainalFirdaus-301220041-Skripsi"
        )
    except Exception as e:
        print(f"\n[GAGAL] Terjadi kesalahan: {e}")
        print("Pastikan token Anda benar dan memiliki izin akses 'repo'.")


if __name__ == "__main__":
    main()
