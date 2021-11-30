# Validation
可將多邊形方框與原始標記進行重疊率計算，透過重疊方框計算混淆矩陣，最後再由混淆矩陣進行指標計算。可將計算後指標與標記覆蓋率直接匯制在原圖上

<h3>需要參數:</h3>

 - name : 圖片名稱(含路徑)。
 - json_name : 該圖片搭配的 JSON 名稱(含路徑)，計算正解用。
 - image : 圖片 array。
 - polygon_list : 分群後的多邊形陣列。
---

<h2>API</h2>

---
- <h4>GET -可以取得的資源</h4>

  - get_confusion_matrix() : 取得混淆舉證。
  - get_coincidence_rate() : 取得預判與正解的覆蓋率與繪製座標( 可以使用 draw_coincidence_rate_to_image() 來繪製)。
  - get_yolo_point() : 取得將原始座標成yolo座標
  
- <h4>DRAW -可繪製資源(因為會自動存圖片所以需要給要儲存的圖片名稱)</h4>

  - draw_coincidence_rate_to_image(save_name) : 繪製正解的覆蓋率在圖片上，會自動透過save_name來存圖片。
  - draw_confusion_matrix(save_name) : 繪製recall、precision、F1_source、probability、color_box在圖片上，會自動透過save_name來存圖片。


