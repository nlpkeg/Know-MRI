import copy
from util.constant import key_attribution, key_neuron_index, key_origin_data
from methods import method_name2sub_module
import util.constant
from util.model_tokenizer import get_cached_model_tok

# reference from echarts:
# https://echarts.apache.org/examples
# https://echarts.apache.org/examples/zh/editor.html?c=scatter-aqi-color

option_template_neuron_attribution = {
  "color": ['blue'],
  "title": {
    "text": 'Contribution and Meaning of the neuron',
    "subtext": 'Data from method: method_name'
  },
  "grid": {
    "left": '50',
    "right": '60',
    "top": '22%',
    "bottom": '15%'
  },
  "xAxis": {
    "type": 'value',
    "name": 'neuron ix',
    "nameGap": 0,
    "nameTextStyle": {
      "fontSize": 16,
      "align": 'right',
      "verticalAlign": 'top',
      "padding": [30, 0, 0, 0], # the top padding will shift the name down so that it does not overlap with the axis-labels
    },
    "nameLocation": 'end',
    "max": 6400,
    "splitLine": {
      "show": False
    }
  },
  "yAxis": {
    "type": 'value',
    "name": 'layer',
    "nameLocation": 'end',
    "nameGap": 20,
    "nameTextStyle": {
      "fontSize": 16
    },
    "max": 48,
    "splitLine": {
      "show": False
    }
  },
"visualMap": [
    {
    "right": "0",
    "top": "10%",
    "dimension": 2,
    "min": 0,
    "max": 15,
    "itemWidth": 10,
    "itemHeight": 50,
    "precision":0.1,
    "text": ["Weight"],
    "textGap": 30,
    "inRange": { 
        "colorLightness": [0.9, 0.1],
        "symbolSize": [5, 10]
    },
    "outOfRange": { "color": ["blue"] },
    "controller": {
        "inRange": { "color": ["blue"] },
        "outOfRange": { "color": ["red"] }
    }
    }
],
  "series": [
    {
      "type": 'scatter',
      "data": []
    }
  ]
}

option_template_hiddenstates = {
  "color": ['blue'],
  "title": {
    "text": 'Attention Weights',
    "subtext": 'layer: {layer}, head: {head}'
  },
  "grid": {
    "left": '50',
    "right": '60',
    "top": '22%',
    "bottom": '15%'
  },
  "xAxis": {
    "type": 'category',
    "data": ["a", "list", "of", "category"],
    "splitArea": {
      "show": True
    }
  },
  "yAxis": {
    "type": 'category',
    "data": ["a", "list", "of", "category"],
    "splitArea": {
      "show": True
    }
  },
"visualMap": [
    {
    "right": "0",
    "top": "10%",
    "dimension": 2,
    "min": 0,
    "max": 15,
    "itemWidth": 10,
    "itemHeight": 50,
    "precision":0.1,
    "text": ["Attribution"],
    "textGap": 30,
    "inRange": { 
        "colorLightness": [0.9, 0.1],
        "symbolSize": [5, 10]
    },
    "outOfRange": { "color": ["blue"] },
    "controller": {
        "inRange": { "color": ["blue"] },
        "outOfRange": { "color": ["red"] }
    }
    }
],
  "series": [
    {
      "type": 'heatmap',
      "data": []
    }
  ]
}

def get_echarts_for_hiddenstates(result, method_name, model_name):
    options = []
    count = 0
    max_fn = len(result[key_origin_data][util.constant.gaol_layer_head_key]) - 1
    for layer, head in result[key_origin_data][util.constant.gaol_layer_head_key]:
      attention_weight = result[key_origin_data]["attention_weight"][layer][head]
      data = [[i, j, attention_weight[i][j]] for i in range(len(attention_weight)) for j in range(len(attention_weight[0]))]
      tokens = [token.replace("<s>", "bos") for token in result[key_origin_data]["tokens"]]
      # instantiate one option
      option = copy.deepcopy(option_template_hiddenstates)
      option["title"]["subtext"] = option["title"]["subtext"].format(layer=layer, head=head)
      option["title"]["subtext"] = option["title"]["subtext"].replace("method_name", method_name)
      option["series"][0]["data"] = data
      # attribution range
      min_value = min(row[2] for row in data)
      max_value = max(row[2] for row in data)
      option["visualMap"][0]["min"] = min_value
      option["visualMap"][0]["max"] = max_value
      # axis range
      option["xAxis"]["data"] = tokens
      option["yAxis"]["data"] = tokens
      des = "" if count < max_fn else "We are visualizing the attention weights of some heads in the model."
      count += 1
      options.append({"interpret_class": util.constant.hiddenstates ,"option":option, "des": des})
    return options

def get_echarts_for_neuron_attribution(result, method_name, model_name):
    # 交互显示top1000的
    topN_token2relative_tokens = {d["Top neurons"]: d["Corresponding top tokens"] for d in result["table"][0]["table_list"]}
    data = []
    for layer, neuron_ix in  result[key_origin_data][key_neuron_index]:
        relative_tokens = topN_token2relative_tokens.get(f"L{layer}.U{neuron_ix}", list())
        item = [neuron_ix, layer, float(result[key_origin_data][key_attribution][layer][neuron_ix]), ", ".join(relative_tokens)]
        data.append(item)
    data = sorted(data, key=lambda d:d[2])
    # instantiate one option
    option = copy.deepcopy(option_template_neuron_attribution)
    option["title"]["subtext"] = option["title"]["subtext"].replace("method_name", method_name)
    option["series"][0]["data"] = data
    attributions = [d[2] for d in data]
    # attribution range
    min_value = min(attributions)
    max_value = max(attributions)
    option["visualMap"][0]["min"] = min_value
    option["visualMap"][0]["max"] = max_value
    # axis range
    mt = get_cached_model_tok(model_name=model_name)
    option["xAxis"]["max"] = mt.num_neurons_perlayer
    option["yAxis"]["max"] = mt.num_layers
    top_neuron_name = [d["Top neurons"] for d in result["table"][0]["table_list"]][:5]
    return {"interpret_class": util.constant.neuron_attribution ,"option":option, "des": "This image represents the contribution scores of positioning neurons. The x-axis represents the layer in which the neuron is located, and the y-axis represents the index of the neuron.",
            "res": f"Top 5 attribution neurons are: {top_neuron_name}"}

def get_echarts_info_with_result(result, method_name, model_name):
    echarts = []
    interpret_class = method_name2sub_module[method_name].interpret_class
    if interpret_class == util.constant.neuron_attribution:
        option = get_echarts_for_neuron_attribution(result,
                                 method_name=method_name,
                                 model_name=model_name)
        echarts.append(option)
    elif interpret_class == util.constant.attention:
        options = get_echarts_for_hiddenstates(result,
                                 method_name=method_name,
                                 model_name=model_name)
        echarts.extend(options)
    return echarts
    



