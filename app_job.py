import os
import time
import classificando


def monitor_folder(folder_path):
    files_set = set()

    while True:
        files = set(os.listdir(folder_path))

        new_files = files - files_set
        if new_files:
            classificando.classificar_texto(new_files)

            print("Novo arquivo recebido!")

        files_set = files
        # Verificação a cada 5 segundos (ajuste conforme necessário)
        time.sleep(5)


if __name__ == "__main__":
    folder_to_monitor = "temp"  # Substitua pelo caminho da pasta que você deseja monitorar
    monitor_folder(folder_to_monitor)
