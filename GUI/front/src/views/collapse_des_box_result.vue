<!-- -----------------------------------------------模板区域--------------------------------------- -->
<template>

    <el-collapse  v-model="activeNames" @change="handleCollapseChange">
        <el-collapse-item name="1">
            <template slot="title">
                <h3 class="item_title el-icon-link">&nbsp;&nbsp;{{ copy_method_name }}</h3>
            </template>
            <div class="item_des_content" >

                <!-- 渲染文本 -->
                <div class="item_des_content_title">Diagnosis result</div>

                <div class="item_des_content_str" v-if="copy_result_method_data['text'] != ''" v-text="copy_result_method_data['text']" style="overflow-wrap: break-word;  margin-bottom: 20px;"></div>

                <!-- 渲染图表 -->
                <div v-for="(item, index) in  copy_result_method_data['echarts']" style="width: 100%;" >
                    <div :ref=index :id=index+copy_method_name :style="{  width: '100%', height: '400px' }"></div>
                    <div v-if="'des' in item && item['des'] != '' " class="item-des" v-text="item['des']"></div>
                    <div v-if="'res' in item && item['res'] != '' " class="item-res" v-text="item['res']"></div>
                </div>

                <!-- 渲染图片 -->
                <div v-for="(item, index) in  copy_result_method_data['imgs']">
                    <div v-text="item['img_name']" align="center" style="font-size: 1.2rem; font-weight: 600;  margin-bottom: 8px;" ></div>
                    <img class="item_des_content_img" :src="item['img_base64']"></img>
                    <div v-if="'des' in item && item['des'] != '' " class="item-des" v-text="item['des']"></div>
                    <div v-if="'res' in item && item['res'] != '' " class="item-res" v-text="item['res']"></div>
                </div>


                <!-- 渲染table -->
                <div v-for="(item, index) in  copy_result_method_data['table']" class="user_tables">
                    <div v-text="item['table_name']" align="center" style="font-size: 1.2rem; font-weight: 600; margin-bottom: 8px;"  ></div>
                    <el-table height="300"
                        :data="item['table_list']"
                        style="width: 100%;margin-bottom: 20px;"
                        row-key="id"
                        border
                        default-expand-all
                        :tree-props="{children: 'children', hasChildren: 'hasChildren'}">

                        <!-- 根据 item 动态渲染 table 列-->
                        <el-table-column
                            v-for="(value, key) in item['table_list'][0]"
                            :key="key"
                            :prop="key"
                            :label="key">
                            <template v-slot="scope">
                                <div class="custom-cell">{{ scope.row[key] }}</div>
                            </template>
                        </el-table-column>

                    </el-table>
                    <div v-if="'des' in item && item['des'] != '' " class="item-des" v-text="item['des']"></div>
                    <div v-if="'res' in item && item['res'] != '' " class="item-res" v-text="item['res']"></div>
                </div>

            </div>
        </el-collapse-item>
    </el-collapse> 



</template>

<!-- ------------------------------------------------脚本区域--------------------------------------- -->
<script>
import * as echarts from "echarts";

export default {
    props: {
        root_method_name: null,
        root_type_name:null,
        method_name: null,
        result_data: null,
    },
   data() {
       return {
            copy_root_method_name: "",
            copy_method_name: "",
            copy_result_method_data: {},

            activeNames:[],
       };
   },

    watch: {

        result_data: {
            handler(newValue, oldValue) {
                this.$nextTick(() => {
                    try {
                        this.init_echarts(newValue);
                    } catch (error) {
                        
                    }
                });
            },
            deep: true
        }

    },
        
    mounted() {
        this.copy_method_name = "" + this.method_name;
        this.copy_root_method_name = "" + this.root_method_name;
        this.copy_result_method_data = JSON.parse(JSON.stringify(this.result_data));
        // console.log("🚀 -> this.copy_result_method_data:\n", this.copy_result_method_data)
    },
    methods: {
        handleCollapseChange() {
            this.$nextTick(() => {
                this.init_echarts(this.result_data);
            });
        },

        init_echarts(newValue) {
            this.copy_result_method_data = JSON.parse(JSON.stringify(newValue))

            // 渲染图表
            for (let i in this.copy_result_method_data['echarts']) {

                // neuron attribution
                if (this.copy_result_method_data['echarts'][i]['interpret_class'] == 'Neuron Attribution') {
                    let param_option = this.copy_result_method_data['echarts'][i]['option'];
                    // 调整默认样式
                    param_option['tooltip'] = {
                        formatter: function (param) {
                            let value = param.value;
                            return ''
                                + 'neuron ix' + '：' + value[0] + '<br>'
                                + 'layer' + '：' + value[1] + '<br>'
                                + 'attribution' + '：' + value[2] + '<br>'
                                + 'relative tokens' + '：' + value[3] + '<br>';
                        }
                    };
                    for (let j in param_option['series']) {
                        param_option['series'][j]['itemStyle'] = {
                            opacity: 0.8,
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowOffsetY: 0,
                            shadowColor: 'rgba(0,0,0,0.3)'
                        }
                    }
                    let chartDom = document.getElementById(i + this.copy_method_name);
                    if (chartDom != null) {
                        let mychart = echarts.getInstanceByDom(chartDom);
                        // 如果实例存在，销毁它
                        if (mychart) {
                            echarts.dispose(chartDom);
                            // mychart.clear();
                        }
                        // 重新初始化图表
                        mychart = echarts.init(chartDom);
                        mychart.setOption(param_option);// 渲染页面
                        mychart.resize();
                        //随着屏幕大小调节图表
                        window.addEventListener("resize", () => {
                            mychart.resize();
                        });
                        // 为图表元素 添加事件
                        mychart.on('click', function (params) {
                            // console.log(mychart.getOption().series[0].data);
                            if (params.componentType == 'series') {
                                let series_data = params.data;
                                let series_data_index = params.dataIndex;
                                // console.log("🚀 -> series_data_index:\n", series_data_index)
                                // console.log("🚀 -> series_data:\n", series_data)
                            }

                        });


                    }

                } else if (this.copy_result_method_data['echarts'][i]['interpret_class'] == 'Hiddenstates') {

                    let param_option = this.copy_result_method_data['echarts'][i]['option'];

                    // 调整默认样式
                    param_option['tooltip'] = {
                        formatter: function (param) {
                            // console.log("🚀 -> param_option:\n", param_option)

                            let x_list = param_option.xAxis.data;
                            let y_list = param_option.yAxis.data;

                            let value = param.value;
                            return ''
                                + 'sourse token' + '：' + x_list[value[0]] + '<br>'
                                + 'target token' + '：' + y_list[value[1]] + '<br>'
                                + 'attention weight' + '：' + value[2] + '<br>'
                        }
                    };
                        
                    for (let j in param_option['series']) {
                        param_option['series'][j]['itemStyle'] = {
                            opacity: 0.8,
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowOffsetY: 0,
                            shadowColor: 'rgba(0,0,0,0.3)'
                        }
                    }

                    let chartDom = document.getElementById(i + this.copy_method_name);
                    if (chartDom != null) {
                        let mychart = echarts.getInstanceByDom(chartDom);
                        // 如果实例存在，销毁它
                        if (mychart) {
                            echarts.dispose(chartDom);
                            // mychart.clear();
                        }
                        // 重新初始化图表
                        mychart = echarts.init(chartDom);
                        mychart.setOption(param_option);// 渲染页面
                        mychart.resize();
                        //随着屏幕大小调节图表
                        window.addEventListener("resize", () => {
                            mychart.resize();
                        });
                        // 为图表元素 添加事件
                        mychart.on('click', function (params) {
                            // console.log(mychart.getOption().series[0].data);
                            if (params.componentType == 'series') {
                                let series_data = params.data;
                                let series_data_index = params.dataIndex;
                                // console.log("🚀 -> series_data_index:\n", series_data_index)
                                // console.log("🚀 -> series_data:\n", series_data)
                            }

                        });
                    }

                } else {
                    // 默认使用传入的option 渲染
                    let param_option = this.copy_result_method_data['echarts'][i]['option'];
                    let chartDom = document.getElementById(i + this.copy_method_name);
                    if (chartDom != null) {
                        let mychart = echarts.getInstanceByDom(chartDom);
                        if (mychart) {
                            echarts.dispose(chartDom);
                        }
                        mychart = echarts.init(chartDom);
                        mychart.setOption(param_option);
                        mychart.resize();
                        window.addEventListener("resize", () => {
                            mychart.resize();
                        });
                    }

                }

            }
            
        }

    },
};
</script>


<!-- ------------------------------------------------样式区域--------------------------------------- -->
<style scoped>

.item_des_content {
	width: 100%;
	/* min-height: 400px; */
}
.item_des_content img {
    display: block; /* 避免默认的行内样式影响 */
    margin: 0 auto; /* 水平居中 */
}
.item_des_content
/* ele */
.el-cascader-panel {
	border: none !important;
}

.item_des_content_img {
    width: calc(100% - 100px);
    height: auto;
}
.item_des_content_title {
    margin-top: 5px;
    margin-left: 5px;
    font-weight: 600;
    font-size: 1.1rem;
    color: #575757;
    /* white-space: pre-wrap;  */
    /* word-wrap: break-word;  */
    /* overflow-y: auto; */
    /* padding: 20px; */
    /* color:#008B8B; */
}


.item_des_content_str {
    white-space: pre-wrap; /* 保留空白符样式 */
    word-wrap: break-word; /* 如果内容太长，允许折行 */
    overflow-y: auto;
    padding: 20px;
    color:#008B8B;
    font-size: 15px;
}

.item-des {
    font-size: 14px;
    margin: 0 auto; /* 水平居中 */
    color:#2882AA;
    margin-top: 10px;
    /* margin-bottom: 20px; */
    width: calc(100% - 100px);
    max-width: calc(100% - 100px);
    white-space: pre-wrap; /* 保留空白符样式 */
    word-wrap: break-word; /* 如果内容太长，允许折行 */
}

.item-res {
    font-size: 14px;
    margin: 0 auto; /* 水平居中 */
    color:#2882AA;
    margin-top: 10px;
    margin-bottom: 20px;
    width: calc(100% - 100px);
    max-width: calc(100% - 100px);
    white-space: pre-wrap; /* 保留空白符样式 */
    word-wrap: break-word; /* 如果内容太长，允许折行 */
    
}
/* el-table 调整 背景透明 */

.custom-cell {
    white-space: pre-wrap; /* 保留空白符样式 */
    word-wrap: break-word; /* 如果内容太长，允许折行 */
}

.user_tables{
   /* width: 50%; */
   margin: auto;
   white-space: pre;
}
.user_tables >>> .el-table--fit{
    padding: 0px;
    border: 1px solid #dcdfe6;
}
.user_tables  >>>  .el-table, .el-table__expanded-cell {
    background-color: transparent;
    border: 1px solid #dcdfe6;
}

.user_tables  >>> .el-table th {
    background-color: transparent!important;
    border: 1px solid #dcdfe6;
}

.user_tables  >>> .el-table tr {
    background-color: transparent!important;
    border: 1px solid #dcdfe6;
}

.user_tables >>>  .el-table--enable-row-transition .el-table__body td, .el-table .cell{
   background-color: transparent;
   border: 1px solid #dcdfe6;
}

</style>