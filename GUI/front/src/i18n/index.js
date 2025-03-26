import Vue from 'vue';
import VueI18n from 'vue-i18n';
import locale from 'element-ui/lib/locale';
Vue.use(VueI18n);
import zh from './config/zh';
import en from './config/en';

import zhLocale from 'element-ui/lib/locale/lang/zh-CN';
import enLocale from 'element-ui/lib/locale/lang/en';

const messages = {
    en: {
      ...en,
      ...enLocale
    },
    zh: {
      ...zh,
      ...zhLocale
    },
}  
const i18n = new VueI18n({
    locale: localStorage.getItem('locale') || 'en', 
    messages:messages,
})
const translate = (localeKey) => {
    const locale = localStorage.getItem("language") || "zh"
    const hasKey = i18n.te(localeKey, locale)  
    const translatedStr = i18n.t(localeKey) 
    if (hasKey) {
        return translatedStr
    }
    return localeKey
}

locale.i18n((key, value) => i18n.t(key, value))
export {
    i18n,
    translate
};