<!-- -----------------------------------------------模板区域--------------------------------------- -->
<template>
    <div class="root">

        <div class="root-top">
            <!-- 输入框 -->
            <div style="display: flex; flex-direction: row; align-items: center;">
                <!-- <div class="item_input"  v-text="input_value" ></div> -->

                <!--  type="textarea" :rows="1" -->
                <el-input v-if="dataset_flag == 'USEREDITINPUT'" v-model="user_input"
                    placeholder="Please enter your question" @input="handleChangeInputValue('')"
                    style="width: calc(100% - 400px); margin-left: 10px; flex: 1;"></el-input>
                <el-select v-if="dataset_flag == 'USEREDITINPUT'" v-model="input_value" clearable default-first-option
                    placeholder="Please enter your question" @change="changeInputValue('')" no-data-text="No Data"
                    style="flex: 1; margin-left: 15px;">
                    <el-option v-for="item in input_list" :key="item['prompt']" :label="item" :value="item">
                    </el-option>
                </el-select>

                <el-select v-else v-model="input_value" clearable default-first-option
                    placeholder="Please enter your question" @change="changeInputValue('')" no-data-text="No Data"
                    style="flex: 1;">
                    <el-option v-for="item in input_list" :key="item" :label="item" :value="item"> </el-option>
                </el-select>

                <el-button v-if="dataset_flag == 'USEREDITINPUT'" type="primary" icon="el-icon-question" round
                    style="margin-left: 15px; max-height: 44.44px;flex: 1;max-width: 130px;"
                    @click="do_search_topn_by_input" :loading="SearchingLoading">Search</el-button>

                <!-- Another random question -->
                <el-button v-if="dataset_flag != 'USEREDITINPUT'" type="primary" icon="el-icon-question" round
                    @click="get_dataset_random" style="flex: 1;margin-left: 15px; max-width: 130px;">Search</el-button>

                <el-button type="primary" icon="el-icon-s-promotion" round
                    style="margin-left: 15px; max-height: 44.44px;flex: 1; max-width: 130px;" @click="do_generate"
                    :loading="GenerateLoading">Diagnose</el-button>

            </div>


        </div>

        <div class="root-bottom">
            <div class="root-left">
                <div class="item_select_datasets">


                    <div class="item-dataset" id="USEREDITINPUT" @click="get_dataset_data_topn('USEREDITINPUT')">
                        <div class="dataset-title">
                            Custom Input
                        </div>
                        <div class="dataset-des">
                            Do not select any dataset, enter a new data entry.
                        </div>
                    </div>

                    <div class="item-dataset" :id="item['name']" v-for="item in dataset_list"" @click="
                        get_dataset_data_topn(item['name'])">
                        <div class="dataset-title">
                            {{ item['name'] }}
                        </div>
                        <div class="dataset-des">
                            {{ item['des'] }}
                        </div>
                    </div>

                </div>
                <div class="item_select_models">
                    <el-form ref="form" label-width="90px" style="margin-left: 0px;">

                        <el-form-item label="Model Type">
                            <el-select v-model="select_model_type" placeholder="Please select"
                                @change="changeSelectModelPath" no-data-text="No Data">
                                <el-option v-for="item in model_config_list" :key="item['model_type']"
                                    :label="item['model_type']" :value="item['model_type']">
                                </el-option>
                            </el-select>
                        </el-form-item>

                        <el-form-item label="Model Path">
                            <!-- <el-select v-model="select_model_path" placeholder="Please select"
                                @change="changeSelectModel" allow-create filterable default-first-option
                                no-data-text="No Data">
                                <el-option v-for="item in model_path_list" :key="item['value']" :label="item['value']"
                                    :value="item['value']">
                                </el-option>
                            </el-select> -->

                            <el-autocomplete class="inline-input" v-model="select_model_path" @input="changeSelectModel"
                                :fetch-suggestions="querySearch" placeholder="Please Input Path">
                                <i class="el-icon-edit el-input__icon" slot="suffix">
                                </i>
                            </el-autocomplete>


                        </el-form-item>

                        <el-form-item label="Method">

                            <el-cascader v-model="select_method" placeholder="Please select" :options="method_list"
                                :props="props">

                                <template v-slot:empty>
                                    No Data
                                </template>

                            </el-cascader>

                        </el-form-item>

                        <el-form-item label="">
                            <el-button type="primary" icon="el-icon-s-cooperation" round @click="load_model"
                                :loading="loading_model_flag">Load Model</el-button>

                        </el-form-item>



                    </el-form>

                </div>

                <div class="input-params">


                </div>

            </div>

            <div class="root-right">

                <div>
                    <!-- 输入行 -->

                    <!-- 返回结果框 -->
                    <div class="item_result" v-text="result['result']"></div>

                </div>

                <div style="display: flex; height: calc(100% - 90px); width: 100%;">

                    <!-- 动态展示结果 -->
                    <div v-if="isEmptyObject_result(result)" class="item_des">
                        <el-empty :image-size="300" description="No Data"></el-empty>
                    </div>

                    <div v-else class="item_des_parent">

                        <div v-for="(root_item, root_index) in result" :key="root_index" v-if="root_index != 'result'"
                            style="width: 100%; height: 100%; display: flex;">
                            <div class="item_des">
                                <h1 class="el-icon-chat-dot-square item_root_title">&nbsp;{{ root_index }}</h1>

                                <div class="item_des_type" v-for="(root_item_type, root_index_type) in root_item"
                                    :key="root_index_type">
                                    <h2 class="el-icon-s-opportunity item_root_type_title">&nbsp; {{ root_index_type }}
                                    </h2>
                                    <collapse_des_box_result v-for="(item, index) in root_item_type" :key="index"
                                        :root_method_name="root_index" :root_type_name="root_index_type"
                                        :method_name="index" :result_data="item"></collapse_des_box_result>
                                </div>

                            </div>

                        </div>

                    </div>

                </div>

            </div>

        </div>

    </div>
</template>

<!-- ------------------------------------------------脚本区域--------------------------------------- -->
<script>
// @ 代表从 src 目录开始
import { axios_instance,getServerIp  } from "@/axios/index";
import collapse_des_box_result from "@/views/collapse_des_box_result.vue";

export default {
    name: "temp",
    components: { collapse_des_box_result },
	data() {
        return {
            // 配置 / flag
            props: { multiple: true },
            SearchingLoading:false,
            GenerateLoading: false,
            loading_model_flag:false,
            // 列表与选择数据
            model_config_list: [],
            model_path_list: [],
            method_list: [],
            dataset_flag:"",
            select_dataset: "",
            select_model_type: "",
            select_model_path: "",
            select_method: [],
            // 用户输入
            user_input: "",
            user_input_search_list:[],
            user_select_obj: {},
            // 右侧展示
            input_value: "",
            input_obj: {
                "prompt":""
            },
            result: this.init_result_obj(true),
            // 之前的
            dataset_list: [],
            input_list: [],
            input_obj_list: [],
            USEREDITINPUT_params : {
                "ground_truth":""
            },
		};
	},
	watch: {},
    mounted() {

        // this.get_model_list();

        // 获取数据集列表  并默认选择数据集 
        axios_instance
            .get("/getDatasetList")
            .then((response) => {
                this.dataset_list = response.data.data;
                // this.select_dataset = this.dataset_list[0].name;
                // this.dataset_flag = "SELECT"
                this.select_dataset = "USEREDITINPUT";
                this.dataset_flag = "USEREDITINPUT"
                this.$nextTick(() => { 

                    axios_instance
                        .get("/getModelList")
                        .then((response) => {
                            console.log("🚀 -> response.data.data:\n", response.data.data)
                        })
                        .catch((error) => { });

                    this.get_dataset_data_topn(this.select_dataset);
                    this.get_model_list();
                })
            })
            .catch((error) => {});

    },
    methods: {

        //  判断是否是空对象
        isEmptyObject(obj) {
            return Object.keys(obj).length === 0;
        },

        //  判断result 是否是空对象 (去除 result字段后)
        isEmptyObject_result(obj) {
            let temp = JSON.parse(JSON.stringify(obj));
            if ("result" in temp) {
                delete temp['result'];
            }
            return Object.keys(temp).length === 0;
        },

        init_result_obj(init_reult_flag) {
            let obj = {
                // result: "",
                // ...
            }

            if (init_reult_flag == false) {
                obj['result'] = this.result['result'];
            } 
            return obj;
        },

        // 获取模型列表 并默认选择模型
        get_model_list() {
            axios_instance
                .get("/getModelSecList")
                .then((response) => {
                    this.model_config_list = response.data.data;
                    this.select_model_type = this.model_config_list[0]['model_type'];
                    
                    for (let i in this.model_config_list) {
                        if (this.model_config_list[i]['model_type'] == this.select_model_type) {
                            this.model_path_list = this.model_config_list[i]['model_list']
                            this.select_model_path = this.model_path_list[0]['value']
                        }
                    }
                    
                    // 初始化 方法列表 并默认选择方法
                    this.changeSelectModel();
                })
                .catch((error) => { });
        },

        // 加载模型
        load_model(){
            let model = this.select_model_type + "/" + this.select_model_path;
            this.loading_model_flag = true;
            let hold = this.$message({
                message: 'The model is loading, please wait a moment ...',
                type: 'warning',
                duration:0
            });
            axios_instance
                .get("/loadModelByName?modelName=" + model)
                .then((response) => {
                    this.loading_model_flag = false;
                    hold.close();
                    this.$message({
                        message: 'The model has loaded successfully!',
                        type: 'success',
                        duration:2000
                    });

                })
                .catch((error) => {
                    this.loading_model_flag = false;
                    hold.close();
                    this.$message({
                        message: 'The model has loaded Error!',
                        type: 'error',
                        duration:2000
                    });
                });
        },

        handleChangeInputValue(value) {
            // console.log("🚀 -> value:\n", value)
            this.user_input_search_list =[];
            this.input_obj_list = [];
            this.input_list = [];
            this.input_obj = {}
            this.input_value = "";
            this.select_dataset = "";
            this.changeInputValue('');
            
        },

        // Search 根据用户输入 获取topn
        do_search_topn_by_input() {

            let value = this.user_input;
            // console.log("🚀 -> value:\n", value)

            if (value == "") {
                this.user_input_search_list = [];
                return;
            }

            let params = {
                "input_text":value,
            }
            this.SearchingLoading = true;
            axios_instance
                .post("/searchTopnByInput",params)
                .then((response) => {
                    this.user_input_search_list = response.data.data;
                    this.input_obj_list = response.data.data;
                    this.input_list = response.data.data.map(item => {
                        return item['prompt'];
                    });
                    if (this.input_list.length > 0) {
                        this.input_obj = this.input_obj_list[0];
                        this.input_value = this.input_list[0];
                        this.changeInputValue('');
                    } else {
                        this.input_obj = {}
                        // this.input_value = "";
                        this.changeInputValue('');
                    }
                    this.SearchingLoading = false;

                })
                .catch((error) => { });
            
        },

        // 获取第一个叶子节点路径
        findFirstLeafNodeWithPath(nodes, path = []) {
            for (const node of nodes) {
                // 更新当前路径
                const currentPath = [...path, node.value];

                // 如果当前节点没有子元素，返回该节点和路径
                if (!node.children) {
                    // return { node, path: currentPath };
                    return currentPath
                } else {
                    // 否则递归地在子元素中查找
                    const result = this.findFirstLeafNodeWithPath(node.children, currentPath);
                    if (result) {
                        return result;
                    }
                }
            }
            return []; 
        },

        querySearch(queryString, cb) {
            for (let i in this.model_config_list) {
                if (this.model_config_list[i]['model_type'] == this.select_model_type) {
                    this.model_path_list = this.model_config_list[i]['model_list'];
                    // this.select_model_path = this.model_path_list[0]['value'];
                }
            }
            cb(this.model_path_list);
        },

        changeSelectModelPath(){

            for (let i in this.model_config_list) {
                if (this.model_config_list[i]['model_type'] == this.select_model_type) {
                    this.model_path_list = this.model_config_list[i]['model_list'];
                    this.select_model_path = this.model_path_list[0]['value'];
                }
            }

        },

        // // 根据模型 获取方法
        changeSelectModel() {
            let model = this.select_model_type + "/" + this.select_model_path;
            // console.log("🚀 -> model:\n", model)
            if (this.select_dataset == "") {
                this.method_list = [];
                this.select_method = [];
                return;
            }
            if (model == "") {
                this.method_list = [];
                this.select_method = [];
                return;
            }

            axios_instance
            	.get("/getMethodListByModelName?model_name=" + model + "&dataset_name=" + this.select_dataset)
            	.then((response) => {
                    this.method_list = response.data.data;
                    // console.log("🚀 -> this.method_list:\n", this.method_list)
                    if (this.method_list.length > 0) {
                        let default_method = this.findFirstLeafNodeWithPath(this.method_list);
                        this.select_method = [ [].concat(default_method) ]
                    } else {
                        this.select_method = [];
                    }
            	})
            	.catch((error) => {});
        },
        
		// 获取数据集中的数据前topn
        get_dataset_data_topn(value) {
            // console.log("🚀 -> value:\n", value)

            if (value == "USEREDITINPUT") {
                this.user_input = "What is the capital of China?";
                this.input_value = "What is the capital of China?";
                this.USEREDITINPUT_params['ground_truth'] = "Beijing";
                this.dataset_flag = "USEREDITINPUT"
                this.select_dataset = "";
            } else {
                this.dataset_flag = "SELECT"
                this.select_dataset = value;
            }

            // 变色
            try {
                let dataset_div_list = document.getElementsByClassName("item-dataset");
                for(let i in dataset_div_list){
                    try {
                        dataset_div_list[i].style.backgroundColor = "#F0F2F6"
                    }catch (error) {
                    }
                }
                document.getElementById(value).style.backgroundColor = "#a1c0c0"
            } catch (error) {}

            // 赋值
            // this.input_value = "";
            this.input_obj = {};
            // 切换数据集,清空之前结果
            this.result = this.init_result_obj(true);

            axios_instance
                .get("/getDatasetDataTopnByName?dataset_name=" + this.select_dataset)
                .then((response) => {
                    this.input_obj_list = response.data.data;
                    this.input_list = response.data.data.map(item => {
                        return item['prompt'];
                    });
                    if (this.input_list.length > 0) {
                        this.input_obj = this.input_obj_list[0];
                        this.input_value = this.input_list[0];
                        this.changeInputValue('');
                    } else {
                        this.input_obj = {}
                        // this.input_value = "";
                        this.changeInputValue('');
                    }
                    
                })
                        
                .catch((error) => {});

            // 更新数据集
            if (value == "USEREDITINPUT") {
                this.select_method = [];
                this.method_list = [];
            } else {
                this.changeSelectModel();
            }
            
		},

		// 随机获取一条数据集中的数据
        get_dataset_random() {
            if (this.select_dataset == "") {
                this.$message({
                    message: "Please select the dataset first",
                    type: "warning",
                    duration: 2000,
                });
                return;
            }
            if (this.select_dataset == "") {
                this.$message({
                    message: "There is no more data",
                    type: "warning",
                    duration: 2000,
                });
                return;
            } 

            axios_instance
                .get("/getDatasetDataRandom1ByName?dataset_name=" + this.select_dataset)
                .then((response) => {
                    this.input_obj = response.data.data;
                    this.input_value = response.data.data['prompt'];
                    this.changeInputValue(response.data.data);
                })
                .catch((error) => {});
            
		},

        // 输入数据变更
        changeInputValue(input_obj_param) {

            this.result = this.init_result_obj(true);
            this.input_obj = this.input_obj_list.filter((x) => {
                if (x['prompt'] == this.input_value) {
                    return x;
                }
            })[0];

            // 如果在input_list中没有，则代表是随机数据 或自定义输入数据
            if (this.input_obj == undefined) {
                this.input_obj = input_obj_param;
            }
            
            // 展示除了prompt的其余字段
            // let newObj = Object.fromEntries(Object.entries(this.input_obj).filter(([key, value]) => key !== "prompt"));

            // 展示固定字段
            let allowedFields = ["dataset_name", "ground_truth"];
            let newObj = Object.fromEntries(
                Object.entries(this.input_obj).filter(([key, value]) => allowedFields.includes(key))
            );

            this.result['result'] = JSON.stringify(newObj);
            let tmp_select_dataset = this.input_obj['dataset_name'];
            // console.log("🚀 -> tmp_select_dataset:\n", tmp_select_dataset)
            if (tmp_select_dataset != undefined) {
                this.select_dataset = tmp_select_dataset;
            }
            this.changeSelectModel();
        },
            
        // 开始推理 流式生成
        do_generate() {

            let model = this.select_model_type + "/" + this.select_model_path;
            if (this.input_value == "") {
                this.$message({
                    message: "The input question cannot be empty",
                    type: "warning",
                    duration: 2000,
                });
                return;
            }

            if (model == "") {
                this.$message({
                    message: "The model cannot be empty",
                    type: "warning",
                    duration: 2000,
                });
                return;
            }

            if (this.select_method.length == 0) {
                this.$message({
                    message: "The method list cannot be empty",
                    type: "warning",
                    duration: 2000,
                });
                return;
            }

            let params = {
                "dataset_name": this.select_dataset,
                "model_name": model,
                "method_name": this.select_method,
                "input": this.input_obj,
            }
            console.log("🚀 -> params:\n", params)
            this.GenerateLoading = true;
            let hold = this.$message({
                message: 'The issue is being analyzed, please wait a moment ...',
                type: 'warning',
                duration:0
            });

            this.result = this.init_result_obj(false);

            async function fetchStreamWithPost(url, params,vuecurrent,holdcurrent) {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(params),
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';  // 用于存放数据块

                while (true) {
                    const { done, value } = await reader.read();
                    buffer += decoder.decode(value, { stream: true });
                    
                    let boundary = buffer.indexOf('\n\n');  // 找到双换行符，标记JSON块的结束
                    while (boundary !== -1) {
                        const jsonString = buffer.slice(0, boundary).trim();  // 截取一个完整的JSON字符串
                        buffer = buffer.slice(boundary + 2);  // 余下部分的缓冲区
                        boundary = buffer.indexOf('\n\n');  // 检查是否还有完整的JSON字符串

                        // Base64 解码
                        let decodedBytes = atob(jsonString);
                        let decodedString = decodeURIComponent(escape(decodedBytes));
                        // console.log("🚀 -> decodedString:\n", decodedString)

                        if (decodedString === "END") {
                            console.log("服务器发送了结束标识符。");
                            vuecurrent.GenerateLoading = false;
                            holdcurrent.close();
                            vuecurrent.$message({
                                message: 'success',
                                type: 'success',
                                duration: 3000
                            });
                            break;
                        }
                        try {
                            let temp_res_obj = JSON.parse(decodedString);
                            console.log("🚀 -> temp_res_obj:\n", temp_res_obj);
                            vuecurrent.result = temp_res_obj;
                        } catch (err) {
                            console.error(err);
                        }
                    }

                    if (done) {
                        break;
                    }
                }
            }

            // 调用 fetchStreamWithPost
            getServerIp().then(ip => {
                fetchStreamWithPost(ip + "/do_generate_stream", params,this,hold).catch(err => {
                    console.error(" fetchStreamWithPost error:  ", err);
                    this.GenerateLoading = false;
                    hold.close();
                    this.$message({
                        message: 'server error',
                        type: 'error',
                        duration: 2000
                    });
                });
            }).catch(err => {});

		},

	},
};

</script>

<!-- ------------------------------------------------样式区域--------------------------------------- -->
<style >
.root {
	display: flex;
    flex-direction: column;
	height: calc(100vh - 100px);

}
.root-top {

    width: calc(100% - 30px);

    background-color: #F0F2F6;
	padding: 10px;
	border: solid 2px #E1E2E6;
	border-radius: 20px;
	/* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); */
	/* transition: transform 0.3s, box-shadow 0.3s; */
	/* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1); */
	/* transition: transform 0.3s, box-shadow 0.3s; */
    margin-bottom: 10px;
}

.root-bottom{
    display: flex; 
    flex-direction: row;
    height: calc(100% - 72px);
}

.root-left {
	/* margin-top: 5px; */
	height: calc(100% - 10px);
    width: 340px;
	display: flex;
    flex-direction: column;
}


.item_select_datasets {
    
    overflow-y: auto;
    height: calc(100% - 200px - 10px);
	background-color: #F0F2F6;
	padding: 10px;
	border: solid 2px #E1E2E6;
	border-radius: 20px;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
    display: flex;
    flex-direction: column;
    align-items: center;

}
.item-dataset {
    margin-top: 25px;
    width: 290px;
    height: 90px;
    padding: 8px;
	border: solid 2px #A1C0C0;
	border-radius: 12px;
    background-color: #F0F2F6;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;

}
.item-dataset:hover {
	transform: translateY(-10px);
	box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2), 0 12px 40px rgba(0, 0, 0, 0.2);
}

/* .item-dataset:focus  {
    background-color: #a1c0c0;
} */


.dataset-title {
    padding: 4px;
    height: 23px;
    color:teal;
    font-size: 1.1rem;
    font-weight: 1000;
    overflow-y:auto;

    white-space: pre-wrap; /* 保留换行符、制表符，以及多个空格 */
    word-wrap: break-word; /* 长单词或 URL 将被换行 */
    overflow-wrap: break-word; /* 处理长单词换行的兼容方案 */

}
.dataset-des {
    padding: 0px;
    padding-bottom: 10px;
    margin-top: 5px;
    color:slategray;
    height: 43px;
    width: calc(100% - 5px);
    overflow-y: auto;
    white-space: pre-wrap; /* 保留换行符、制表符，以及多个空格 */
    word-wrap: break-word; /* 长单词或 URL 将被换行 */
    overflow-wrap: break-word; /* 处理长单词换行的兼容方案 */
}


.item_select_models {
    height: 200px;
	background-color: #F0F2F6;
    margin-top: 12px;
	padding: 10px;
	border: solid 2px #E1E2E6;
	border-radius: 20px;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
}

.root-right {
    background-color: #F0F2F6;
    border-radius: 15px;
    padding: 10px;

    margin-left: 8px;
    margin-right: 8px;
	height: calc(100% - 25px);
    width: calc(100% - 340px - 20px);
	display: flex;
    flex-direction: column;
	justify-content: space-around;
}
.item_input {

    min-height: 20px;
    max-height: 45px;
    

	/* background-color: bisque; */
    /* white-space: pre; */
    margin-left: 10px;
    /* width: calc(100% - 415px); */
    width: calc(100% - 180px);
	padding: 10px;
	color: darkcyan;
	/* background-color: darkseagreen; */
	background-color: #F0F2F6;
	font-size: 1.05rem;
	border: solid 0.5px #E1E2E6;
	border-radius: 20px;
    box-shadow: 0 4px 8px rgba(192, 119, 119, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s, box-shadow 0.5s;

    overflow-y: auto;
    white-space: pre-wrap; /* 保留换行符、制表符，以及多个空格 */
    word-wrap: break-word; /* 长单词或 URL 将被换行 */
    overflow-wrap: break-word; /* 处理长单词换行的兼容方案 */

}
.item_input:hover{
    transform: scale(1.01); /* 放大 */
    box-shadow: 0 6px 12px rgba(192, 119, 119, 0.2), 0 10px 30px rgba(0, 0, 0, 0.2); /* 增强阴影效果 */
}


.item_result {
	/* background-color: bisque; */
    /* white-space: pre; */
    margin-left: 10px;
    width: calc(100% - 55px);
	height: 25px;
	/* margin-top: 10px; */
	padding: 15px;
	color: darkcyan;
	/* background-color: darkseagreen; */
	background-color: #F0F2F6;
	font-size: 1.05rem;
	border: solid 0.5px #E1E2E6;
	border-radius: 20px;
    box-shadow: 0 4px 8px rgba(192, 119, 119, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s, box-shadow 0.5s;

    overflow-y: auto;
    white-space: pre-wrap; /* 保留换行符、制表符，以及多个空格 */
    word-wrap: break-word; /* 长单词或 URL 将被换行 */
    overflow-wrap: break-word; /* 处理长单词换行的兼容方案 */


}
.item_result:hover{
    transform: scale(1.01); /* 放大 */
    box-shadow: 0 6px 12px rgba(192, 119, 119, 0.2), 0 10px 30px rgba(0, 0, 0, 0.2); /* 增强阴影效果 */
}

.item_des_parent{
    display: flex;
    height: 100%;
    width: 100%;
}

.item_des {
	flex: 1;
    white-space: pre;
    margin-left: 5px;
    margin-right: 5px;
	padding: 20px;
	overflow-y: auto;
	border-radius: 20px;
	border: solid 3px #E1E2E6;
	box-shadow: 0 4px 8px rgba(192, 119, 119, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1);
	transition: transform 0.3s, box-shadow 0.3s;
    resize: horizontal; 
}
.item_des:hover {
    transform: scale(1.01); 
    box-shadow: 0 6px 12px rgba(192, 119, 119, 0.2), 0 10px 30px rgba(0, 0, 0, 0.2);
}


.item_root_title {
	color: darkcyan;
    font-size: 2rem;
    font-weight: bold;
}

.item_root_type_title {
	color: #F56C6C;
    font-size: 1.35rem;
    font-weight: bold;
}

.item_title {
	color: chocolate;
    font-size: 1.2rem;
    font-weight:520;
}

.el-select .el-input__inner {
	/* height: 80px; */
	border-radius: 20px;
}

.el-collapse-item__wrap{
    background-color: transparent !important;
}
.el-collapse-item__header {
    background-color: transparent !important;
}

.el-form {
    height: 100%;
    overflow-y: auto;
}

.el-textarea__inner{
    margin-top: 10px;
    padding: 15px;
    border-radius: 20px;

}

.el-button.is-round:hover {
    transform: scale(1.01); /* 放大 */
    box-shadow: 0 6px 12px rgba(192, 119, 119, 0.2), 0 10px 30px rgba(0, 0, 0, 0.2); /* 增强阴影效果 */
}
</style>

<!-- flask 启动命令 -->
<!-- /home/liujiaxiang/.conda/envs/inter_fn/bin/python GUI/flask_server.py  -->