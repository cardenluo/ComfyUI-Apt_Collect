
//声明，代码作者是 "yolain" 节点下载地址  "https://github.com/yolain/"）


import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const locale = getLocale()




class Toast{

	constructor() {
		this.info_icon = `<svg focusable="false" data-icon="info-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm32 664c0 4.4-3.6 8-8 8h-48c-4.4 0-8-3.6-8-8V456c0-4.4 3.6-8 8-8h48c4.4 0 8 3.6 8 8v272zm-32-344a48.01 48.01 0 010-96 48.01 48.01 0 010 96z"></path></svg>`
		this.success_icon = `<svg focusable="false" data-icon="check-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z"></path></svg>`
		this.error_icon = `<svg focusable="false" data-icon="close-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" fill-rule="evenodd" viewBox="64 64 896 896"><path d="M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm127.98 274.82h-.04l-.08.06L512 466.75 384.14 338.88c-.04-.05-.06-.06-.08-.06a.12.12 0 00-.07 0c-.03 0-.05.01-.09.05l-45.02 45.02a.2.2 0 00-.05.09.12.12 0 000 .07v.02a.27.27 0 00.06.06L466.75 512 338.88 639.86c-.05.04-.06.06-.06.08a.12.12 0 000 .07c0 .03.01.05.05.09l45.02 45.02a.2.2 0 00.09.05.12.12 0 00.07 0c.02 0 .04-.01.08-.05L512 557.25l127.86 127.87c.04.04.06.05.08.05a.12.12 0 00.07 0c.03 0 .05-.01.09-.05l45.02-45.02a.2.2 0 00.05-.09.12.12 0 000-.07v-.02a.27.27 0 00-.05-.06L557.25 512l127.87-127.86c.04-.04.05-.06.05-.08a.12.12 0 000-.07c0-.03-.01-.05-.05-.09l-45.02-45.02a.2.2 0 00-.09-.05.12.12 0 00-.07 0z"></path></svg>`
		this.warn_icon = `<svg focusable="false" data-icon="exclamation-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm-32 232c0-4.4 3.6-8 8-8h48c4.4 0 8 3.6 8 8v272c0 4.4-3.6 8-8 8h-48c-4.4 0-8-3.6-8-8V296zm32 440a48.01 48.01 0 010-96 48.01 48.01 0 010 96z"></path></svg>`
		this.loading_icon = `<svg focusable="false" data-icon="loading" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="0 0 1024 1024"><path d="M988 548c-19.9 0-36-16.1-36-36 0-59.4-11.6-117-34.6-171.3a440.45 440.45 0 00-94.3-139.9 437.71 437.71 0 00-139.9-94.3C629 83.6 571.4 72 512 72c-19.9 0-36-16.1-36-36s16.1-36 36-36c69.1 0 136.2 13.5 199.3 40.3C772.3 66 827 103 874 150c47 47 83.9 101.8 109.7 162.7 26.7 63.1 40.2 130.2 40.2 199.3.1 19.9-16 36-35.9 36z"></path></svg>`
	}

	async showToast(data){
		let container = document.querySelector(".easyuse-toast-container");
		if (!container) {
			container = document.createElement("div");
			container.classList.add("easyuse-toast-container");
			document.body.appendChild(container);
		}
		await this.hideToast(data.id);
		const toastContainer = document.createElement("div");
		const content = document.createElement("span");
		content.innerHTML = data.content;
		toastContainer.appendChild(content);
		for (let a = 0; a < (data.actions || []).length; a++) {
			const action = data.actions[a];
			if (a > 0) {
				const sep = document.createElement("span");
				sep.innerHTML = "&nbsp;|&nbsp;";
				toastContainer.appendChild(sep);
			}
			const actionEl = document.createElement("a");
			actionEl.innerText = action.label;
			if (action.href) {
				actionEl.target = "_blank";
				actionEl.href = action.href;
			}
			if (action.callback) {
				actionEl.onclick = (e) => {
					return action.callback(e);
				};
			}
			toastContainer.appendChild(actionEl);
		}
		const animContainer = document.createElement("div");
		animContainer.setAttribute("toast-id", data.id);
		animContainer.appendChild(toastContainer);
		container.appendChild(animContainer);
		await sleep(64);
		animContainer.style.marginTop = `-${animContainer.offsetHeight}px`;
		await sleep(64);
		animContainer.classList.add("-show");
		if (data.duration) {
			await sleep(data.duration);
			this.hideToast(data.id);
		}
	}
	async hideToast(id) {
		const msg = document.querySelector(`.easyuse-toast-container > [toast-id="${id}"]`);
		if (msg === null || msg === void 0 ? void 0 : msg.classList.contains("-show")) {
			msg.classList.remove("-show");
			await sleep(750);
		}
		msg && msg.remove();
	}
	async clearAllMessages() {
		let container = document.querySelector(".easyuse-toast-container");
		container && (container.innerHTML = "");
	}

	async copyright(duration = 5000, actions = []) {
		this.showToast({
			id: `toast-info`,
			content: `${this.info_icon} ${$t('Workflow created by')} <a href="https://github.com/yolain/">Yolain</a> , ${$t('Watch more video content')} <a href="https://space.bilibili.com/1840885116">B站乱乱呀</a>`,
			duration,
			actions
		});
	}
	async info(content, duration = 3000, actions = []) {
		this.showToast({
			id: `toast-info`,
			content: `${this.info_icon} ${content}`,
			duration,
			actions
		});
	}
	async success(content, duration = 3000, actions = []) {
		this.showToast({
			id: `toast-success`,
			content: `${this.success_icon} ${content}`,
			duration,
			actions
		});
	}
	async error(content, duration = 3000, actions = []) {
		this.showToast({
			id: `toast-error`,
			content: `${this.error_icon} ${content}`,
			duration,
			actions
		});
	}
	async warn(content, duration = 3000, actions = []) {
		this.showToast({
			id: `toast-warn`,
			content: `${this.warn_icon} ${content}`,
			duration,
			actions
		});
	}
	async showLoading(content, duration = 0, actions = []) {
		this.showToast({
			id: `toast-loading`,
			content: `${this.loading_icon} ${content}`,
			duration,
			actions
		});
	}

	async hideLoading() {
		this.hideToast("toast-loading");
	}

}

export const toast = new Toast();


export function sleep(ms = 100, value) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(value);
        }, ms);
    });
}
export function addPreconnect(href, crossorigin=false){
    const preconnect = document.createElement("link");
    preconnect.rel = 'preconnect'
    preconnect.href = href
    if(crossorigin) preconnect.crossorigin = ''
    document.head.appendChild(preconnect);
}
export function addCss(href, base=true) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.type = "text/css";
    link.href =  base ? "extensions/ComfyUI-Easy-Use/"+href : href;
    document.head.appendChild(link);
}

export function addMeta(name, content) {
    const meta = document.createElement("meta");
    meta.setAttribute("name", name);
    meta.setAttribute('content', content);
    document.head.appendChild(meta);
}

export function deepEqual(obj1, obj2) {
  if (typeof obj1 !== typeof obj2) {
    return false
  }
  if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
    return obj1 === obj2
  }
  const keys1 = Object.keys(obj1)
  const keys2 = Object.keys(obj2)
  if (keys1.length !== keys2.length) {
    return false
  }
  for (let key of keys1) {
    if (!deepEqual(obj1[key], obj2[key])) {
      return false
    }
  }
  return true
}


export function getLocale(){
    const locale = localStorage['AGL.Locale'] || localStorage['Comfy.Settings.AGL.Locale'] || 'en-US'
    return locale
}

export function spliceExtension(fileName){
   return fileName.substring(0,fileName.lastIndexOf('.'))
}
export function getExtension(fileName){
   return fileName.substring(fileName.lastIndexOf('.') + 1)
}

export function formatTime(time, format) {
  time = typeof (time) === "number" ? time : (time instanceof Date ? time.getTime() : parseInt(time));
  if (isNaN(time)) return null;
  if (typeof (format) !== 'string' || !format) format = 'yyyy-MM-dd hh:mm:ss';
  let _time = new Date(time);
  time = _time.toString().split(/[\s\:]/g).slice(0, -2);
  time[1] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'][_time.getMonth()];
  let _mapping = {
    MM: 1,
    dd: 2,
    yyyy: 3,
    hh: 4,
    mm: 5,
    ss: 6
  };
  return format.replace(/([Mmdhs]|y{2})\1/g, (key) => time[_mapping[key]]);
}


let origProps = {};
export const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

export const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

export function updateNodeHeight(node) {node.setSize([node.size[0], node.computeSize()[1]]);}

export function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "easyHidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
}

export function isLocalNetwork(ip) {
  const localNetworkRanges = [
    '192.168.',
    '10.',
    '127.',
    /^172\.((1[6-9]|2[0-9]|3[0-1])\.)/
  ];

  return localNetworkRanges.some(range => {
    if (typeof range === 'string') {
      return ip.startsWith(range);
    } else {
      return range.test(ip);
    }
  });
}


/**
* accAdd 高精度加法
* @since 1.0.10
* @param {Number} arg1
* @param {Number} arg2
* @return {Number}
*/
export function accAdd(arg1, arg2) {
  let r1, r2, s1, s2,max;
  s1 = typeof arg1 == 'string' ? arg1 : arg1.toString()
  s2 = typeof arg2 == 'string' ? arg2 : arg2.toString()
  try { r1 = s1.split(".")[1].length } catch (e) { r1 = 0 }
  try { r2 = s2.split(".")[1].length } catch (e) { r2 = 0 }
  max = Math.pow(10, Math.max(r1, r2))
  return (arg1 * max + arg2 * max) / max
}
/**
 * accSub 高精度减法
 * @since 1.0.10
 * @param {Number} arg1
 * @param {Number} arg2
 * @return {Number}
 */
export function accSub(arg1, arg2) {
  let r1, r2, max, min,s1,s2;
  s1 = typeof arg1 == 'string' ? arg1 : arg1.toString()
  s2 = typeof arg2 == 'string' ? arg2 : arg2.toString()
  try { r1 = s1.split(".")[1].length } catch (e) { r1 = 0 }
  try { r2 = s2.split(".")[1].length } catch (e) { r2 = 0 }
  max = Math.pow(10, Math.max(r1, r2));
  //动态控制精度长度
  min = (r1 >= r2) ? r1 : r2;
  return ((arg1 * max - arg2 * max) / max).toFixed(min)
}
/**
 * accMul 高精度乘法
 * @since 1.0.10
 * @param {Number} arg1
 * @param {Number} arg2
 * @return {Number}
 */
export function accMul(arg1, arg2) {
  let max = 0, s1 =  typeof arg1 == 'string' ? arg1 : arg1.toString(), s2 = typeof arg2 == 'string' ? arg2 : arg2.toString();
  try { max += s1.split(".")[1].length } catch (e) { }
  try { max += s2.split(".")[1].length } catch (e) { }
  return Number(s1.replace(".", "")) * Number(s2.replace(".", "")) / Math.pow(10, max)
}
/**
 * accDiv 高精度除法
 * @since 1.0.10
 * @param {Number} arg1
 * @param {Number} arg2
 * @return {Number}
 */
export function accDiv(arg1, arg2) {
  let t1 = 0, t2 = 0, r1, r2,s1 =  typeof arg1 == 'string' ? arg1 : arg1.toString(), s2 = typeof arg2 == 'string' ? arg2 : arg2.toString();
  try { t1 = s1.toString().split(".")[1].length } catch (e) { }
  try { t2 = s2.toString().split(".")[1].length } catch (e) { }
  r1 = Number(s1.toString().replace(".", ""))
  r2 = Number(s2.toString().replace(".", ""))
  return (r1 / r2) * Math.pow(10, t2 - t1)
}
Number.prototype.div = function (arg) {
  return accDiv(this, arg);
}




const seedNodes = ["easy seed", "easy latentNoisy", "easy wildcards", "easy preSampling", "easy preSamplingAdvanced", "easy preSamplingNoiseIn", "easy preSamplingSdTurbo", "easy preSamplingCascade", "easy preSamplingDynamicCFG", "easy preSamplingLayerDiffusion", "easy fullkSampler", "easy fullCascadeKSampler"]
const loaderNodes = ["easy fullLoader", "easy a1111Loader", "easy comfyLoader", "easy fluxLoader", "easy hunyuanDiTLoader", "easy pixArtLoader"]


app.registerExtension({
	name: "comfy.easyUse.dynamicWidgets",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		function addText(arr_text) {
			var text = '';
			for (let i = 0; i < arr_text.length; i++) {
				text += arr_text[i];
			}
			return text
		}

		if (nodeData.name == 'view_combo') {
			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = async function () {
				onAdded ? onAdded.apply(this, []) : undefined;
				let prompt_widget = this.widgets.find(w => w.name == "prompt")
				const button = this.addWidget("button", "get values from COMBO link", '', () => {
					const output_link = this.outputs[1]?.links?.length>0 ? this.outputs[1]['links'][0] : null
					const all_nodes = app.graph._nodes
					const node = all_nodes.find(cate=> cate.inputs?.find(input=> input.link == output_link))
					if(!output_link || !node){
						toast.error($t('No COMBO link'), 3000)
						return
					}
					else{
						const input = node.inputs.find(input=> input.link == output_link)
						const widget_name = input.widget.name
						const widgets = node.widgets
						const widget = widgets.find(cate=> cate.name == widget_name)
						let values = widget?.options.values || null
						if(values){
							values = values.join('\n')
							prompt_widget.value = values
						}
					}
				}, {
					serialize: false
				})
			}
		}
	}
});


const getSetWidgets = ['rescale_after_model', 'rescale',
						'lora_name', 'lora1_name', 'lora2_name', 'lora3_name', 
						'refiner_lora1_name', 'refiner_lora2_name', 'upscale_method', 
						'image_output', 'add_noise', 'info', 'sampler_name',
						'ckpt_B_name', 'ckpt_C_name', 'save_model', 'refiner_ckpt_name',
						'num_loras', 'num_controlnet', 'mode', 'toggle', 'resolution', 'ratio', 'target_parameter',
	'input_count', 'replace_count', 'downscale_mode', 'range_mode','text_combine_mode', 'input_mode',
	'lora_count','ckpt_count', 'conditioning_mode', 'preset', 'use_tiled', 'use_batch', 'num_embeds',
	"easing_mode", "guider", "scheduler", "inpaint_mode", 't5_type', 'rem_mode'
]

function getSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (getSetWidgets.includes(w.name)) {
				if(node.comfyClass.indexOf("easy XYInputs:") != -1) widgetLogic3(node, w)
				else if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
				else widgetLogic(node, w);
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {
					get() {
						return widgetValue;
					},
					set(newVal) {
						if (newVal !== widgetValue) {
							widgetValue = newVal;
							if(node.comfyClass.indexOf("easy XYInputs:") != -1) widgetLogic3(node, w)
							else if(w.name == 'sampler_name' && node.comfyClass == 'easy preSamplingSdTurbo') widgetLogic2(node, w);
							else widgetLogic(node, w);
						}
					}
				});
			}
		}
}