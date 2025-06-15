const imageInput = document.getElementById('imageInput');
const fileNameDiv = document.getElementById('fileName');
const preview = document.getElementById('preview');
const resultDiv = document.getElementById('result');

// Sidebar toggle
const menuIcon = document.getElementById('menu-icon');
const sidebar = document.getElementById('sidebar');

menuIcon.onclick = () => {
  sidebar.classList.toggle('active');
};

document.addEventListener('click', (e) => {
  if (!sidebar.contains(e.target) && !menuIcon.contains(e.target)) {
    sidebar.classList.remove('active');
  }
});

// Show preview and clear result only when a new file is selected
imageInput.addEventListener('change', function () {
  const file = this.files[0];

  if (file) {
    fileNameDiv.textContent = `📁 Selected file: ${file.name}`;
    
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Do not clear result here — let it persist
  } else {
    fileNameDiv.textContent = '';
    preview.src = '';
    preview.style.display = 'none';
    resultDiv.innerText = '';
  }
});

// Submit form
document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    resultDiv.innerText = "⚠️ Please select an image first.";
    return;
  }

  resultDiv.innerText = "⏳ Processing...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    // Format nicely
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

    let output = '';
for (const [key, value] of Object.entries(data)) {
  if (typeof value === "number") {
    const percent = (value * 100).toFixed(2);
    output += `🔹 ${capitalize(key)}: ${percent}%\n`;
  } else {
    output += `🔹 ${capitalize(key)}: ${value}\n`;
  }
}

    resultDiv.innerText = output.trim();
    console.log("Upload result:", data);
  } catch (error) {
    console.error("Upload error:", error);
    resultDiv.innerText = "❌ Error uploading file.";
  }
});
