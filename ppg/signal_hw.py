def signal_extract(arquivo, freq = 500, dois_ou_quatro = 0):

    sr_dados_red = []
    sr_dados_ir = []
    dados_red = []
    dados_ir = []
    conteudo = []
    fs = freq   #frequencia de amostragem do sinal


    file1 = open(arquivo,'r')
    for linha in file1:
        conteudo.append(linha.rstrip())
    file1.close()

    if(dois_ou_quatro == 0):
        for i in range(len(conteudo)):
            sr_valor_red, sr_valor_ir, valor_red, valor_ir = conteudo[i].split(",")
            sr_dados_red.append(sr_valor_red)
            sr_dados_ir.append(sr_valor_ir)
            dados_red.append(valor_red)
            dados_ir.append(valor_ir)

        for i in range(len(conteudo)):
            sr_dados_red[i] = float(sr_dados_red[i])
            sr_dados_ir[i] = float(sr_dados_ir[i])
            dados_red[i] =float(dados_red[i])
            dados_ir[i] = float(dados_ir[i])
    else: 
        for i in range(len(conteudo)):
            valor_red, valor_ir = conteudo[i].split(",")
            dados_red.append(valor_red)
            dados_ir.append(valor_ir)

        for i in range(len(conteudo)):
            dados_red[i] =float(dados_red[i])
            dados_ir[i] = float(dados_ir[i])


    sred = sr_dados_red
    sir = sr_dados_ir

    redppg = dados_red
    irppg = dados_ir
    
    if (dois_ou_quatro == 0):
        return redppg, irppg, sred, sir
    else:
        return redppg, irppg