import Vue from 'vue'
import App from './App.vue'
import router from './router'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import axios from 'axios';
import { i18n } from "@/i18n/index.js";

Vue.prototype.$axios = axios;
Vue.use(i18n);
Vue.use(ElementUI);

new Vue({
    router,
    i18n,
  render: (h) => h(App)
}).$mount('#app')
