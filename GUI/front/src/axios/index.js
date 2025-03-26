// 引入
import axios from "axios";

export const baseURL = "/";

// 创建请求实例
export const axios_instance = axios.create({
	// axios 前置 url
	baseURL: baseURL,
	// 指定请求超时的毫秒数
	timeout: 0,
	// 表示跨域请求时是否需要使用凭证
	withCredentials: false,
});

// 前置拦截器（发起请求之前的拦截）
axios_instance.interceptors.request.use(
	(config) => {
		/**
		 * 在这里一般会携带前台的参数发送给后台，比如下面这段代码：
		 * const token = getToken()
		 * if (token) {
		 *  config.headers.token = token
		 * }
		 */
		return config;
	},
	(error) => {
		return Promise.reject(error);
	}
);

// 后置拦截器（获取到响应时的拦截）
axios_instance.interceptors.response.use(
	(response) => {
		/**
		 * 根据你的项目实际情况来对 response 和 error 做处理
		 * 这里对 response 和 error 不做任何处理，直接返回
		 */
		return response;
	},
	(error) => {
		const { response } = error;
		if (response && response.data) {
			return Promise.reject(error);
		}
		const { message } = error;
		console.error(message);
		return Promise.reject(error);
	}
);

// 导出常用函数

/**
 * @param {string} url
 * @param {object} data
 * @param {object} params
 */
export const post = (url, data = {}, params = {}) => {
	return axios_instance({
		method: "post",
		url,
		data,
		params,
	});
};

/**
 * @param {string} url
 * @param {object} params
 */
export const get = (url, params = {}) => {
	return axios_instance({
		method: "get",
		url,
		params,
	});
};

// 获取服务器ip的异步函数
export const getServerIp = async () => {
	try {
		const response = await axios_instance.get("/getip");
		return response.data.data;
	} catch (error) {
		console.error("获取IP地址时出错:", error);
		throw error;
	}
};