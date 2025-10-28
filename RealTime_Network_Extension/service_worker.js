// service_worker.js

const API_URL = "http://127.0.0.1:5000/api/check_url"; // Local Flask backend

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !tab.url) return;

  // Send the visited URL to the backend
  fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ url: tab.url })
  })
    .then(response => response.json())
    .then(data => {
      console.log("🔍 Threat check:", data);

      if (!data || !data.result) return;

      // Show a notification if website is harmful
      if (data.result === "attack") {
        chrome.action.setBadgeText({ text: "⚠", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "#ff0000", tabId: tabId });

        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon.png",
          title: "⚠ Suspicious Website Detected!",
          message: `This website (${data.domain}) may be harmful. Proceed with caution.`
        });
      } else {
        chrome.action.setBadgeText({ text: "✔", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "#00a82d", tabId: tabId });
      }
    })
    .catch(err => {
      console.log("❌ Error contacting backend:", err);
      chrome.action.setBadgeText({ text: "!" });
      chrome.action.setBadgeBackgroundColor({ color: "#808080" });
    });
});
