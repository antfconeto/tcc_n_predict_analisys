Dataset de validação Nutrinitro (imagens croppadas 20%)

COMO TRANSFERIR PARA O CELULAR
------------------------------
1. Copie a pasta 'cropped/' para o celular (USB, Drive, adb push, etc.)
2. No app: crie análise de Capim Marandu
3. Importe as imagens da galeria (pasta 18-05, 21-05, 26-05, 01-06)
4. Rode análise: Rede Neural MLP Campeã (nitrogen_mlp)
5. Após concluir: botão DEBUG (ícone de bug) → Copiar JSON
6. Envie o JSON para avaliação de acurácia

ADB (exemplo):
  adb push data/export_app_validation/cropped /sdcard/Download/nutrinitro_validation/

ARQUIVOS
--------
  cropped/{data}/P01.jpg ...  — imagens prontas
  manifest.json               — referência Falker SPAD por parcela
  nutrinitro_validation.zip   — pacote completo para download

REFERÊNCIA
----------
  falker_spad_reference em manifest.json = média Falker (SPAD) por Data+Ponto
