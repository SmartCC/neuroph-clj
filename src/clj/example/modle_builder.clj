(ns example.modle-builder
  (:require [cheshire.core :as che]
            [neuroph-clj.core :as network])
  (:import [org.neuroph.core.transfer Linear Step]
           [org.neuroph.nnet.comp.neuron InputNeuron]
           [org.neuroph.nnet.learning LMS]))

(def stock-url "http://stock2.finance.sina.com.cn/futures/api/json.php/CffexFuturesService.getCffexFuturesMiniKLine5m?symbol=IF1306")

(defn save-stock-data
  "根据url获取股指数据，并使用spit保存（spit与slurp作用相反）"
  [url out-path]
  (che/generate-stream (che/parse-string (slurp url)) (clojure.java.io/writer out-path)))
;  (spit "stock.json" (vec (slurp url))))

(defn read-stock-data
  [file-path]
  (che/parse-stream (clojure.java.io/reader file-path)))

(defn parse-stock-data
  [stock-data]
  (map
   (fn [[d _ _ _ v n]] [d (Double/parseDouble v) (Integer/parseInt n)])
   stock-data))

#_(defn get-stock-price-and-num
  [stock-data]
  (map (fn [[_ d1 _] [_ d2 n]] [(if (pos? (- d2 d1)) 1 0) n])  (rest stock-data) stock-data))

(defn get-stock-price-and-num
  [stock-data]
  (map (fn [[_ d1 _] [_ d2 n]] (if (pos? (- d2 d1)) 1 0))  (rest stock-data) stock-data))

(defn create-input-and-out-seq
  [stock-data seq-length]
  (loop [stock-data stock-data row-data [] expectation []]
    (if (< (count stock-data) (inc seq-length))
      [row-data expectation]
      (recur (rest stock-data) (conj row-data (take seq-length stock-data)) (conj expectation [(nth stock-data seq-length)])))))

(defn test-neural-network
  "测试神经网络模型"
  [*nn* row-data expectation]
  (let [row-data (map double-array row-data)
        expectation (map double-array expectation)]
    (->> (map #(= (vec %2)
                  (vec (network/run-nn *nn* %))) row-data expectation)
         (filter identity)
         count
         (#(* 100 (/ % (count expectation))))
         double)))

(defn run-model
  [stock-data]
  (let [[row-data expectation] (create-input-and-out-seq stock-data 32)
        *training-set* (network/create-data-set 32 (map double-array row-data) 1 (map double-array expectation))
        *layers* (network/create-layers [32 1 1] [{:neuronType InputNeuron :transferFunction Step} {:transferFunction Step} {:transferFunction Step}])
        *nn* (network/create-neural-network network/MULTI-LAYER-PERCEPTRON-TYPE *layers*)
        *learning-rule* (doto (LMS.) (.setMaxIterations 2000))]
    (network/learn *nn* *learning-rule* *training-set*)
    (test-neural-network *nn* row-data expectation)))

;(def d (->> (read-stock-data "stock1.json") parse-stock-data get-stock-price-and-num))
