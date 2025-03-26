


def main():
    final_result = {"s":[[1,0.1],["wo"],[1,0.1],["wo"]],"v":[[1,0.1],["wo"],[1,0.1],["wo"]]}
    original_data = {"sample_tokens":[],"topk_embeddings":[]}
    table = []
    for k,v in final_result.items():
        current_table = {"table_name":f" token {k}","table_list":[],
        "table_des":"Each row represents one of the top-k activation values along with its corresponding dimensional index and the several words from the vocabulary that have the highest activation values in this dimension","table_res":""}
        original_data["sample_tokens"].extend(k)
        current_tabellist = []
        v_values = []
        for i in range(0,len(v),2):
            v_values.append(v[i])
            current_tabellist.append({"dimension":v[i][0],"activation":v[i][1],"tokens with the highest activation value for this dimension in the vocabulary":v[i+1]})
            current_table["table_list"]=current_tabellist
        original_data["topk_embeddings"].append(v_values)
        table.append(current_table)
    
    
    
    ffinal_result ={   
    "origin_data": "any", 
    "table": [{"table_name": "xxx1", "table_list": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "tabel_des": "", "tabel_res": ""}, 
                {"table_name": "xxx2", "table_list": [{"a2": 1, "b2": 2}, {"a2": 3, "b2": 4}], "tabel_des": "", "tabel_res": ""}], 
    "result_des": "The results generated using the SPINE sparse encoder:the top-k activation values and their corresponding tokens with the largest activation values in the respective dimensions" 
}   
    ffinal_result["origin_data"] = original_data
    ffinal_result["table"] = table

    print(ffinal_result)
if __name__ == "__main__":
    main()