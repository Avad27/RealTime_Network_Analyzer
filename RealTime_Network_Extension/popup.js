chrome.storage.local.get(["last_url", "last_result"], (data) => {
    document.getElementById("url").innerText = data.last_url || "No Site Yet";
    document.getElementById("result").innerText = data.last_result || "UNKNOWN";
});
