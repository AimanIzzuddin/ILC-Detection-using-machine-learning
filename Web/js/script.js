const imageInput = document.getElementById('imageInput');
const fileNameDiv = document.getElementById('fileName');
const preview = document.getElementById('preview');

imageInput.addEventListener('change', function () {
  const file = this.files[0];

  if (file) {
    fileNameDiv.textContent = `Selected file: ${file.name}`;
    
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    fileNameDiv.textContent = '';
    preview.src = '';
    preview.style.display = 'none';
  }
});
document.getElementById("uploadForm").onsubmit = async function (e) {
  e.preventDefault();

  const formData = new FormData();
  const fileInput = document.getElementById("image");
  formData.append("file", fileInput.files[0]);

  const res = await fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData
  });

  const result = await res.json();
  document.getElementById("result").innerText = JSON.stringify(result, null, 2);
};
console.log("Upload result:", result);
