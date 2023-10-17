
            
        /* document.addEventListener("DOMContentLoaded", function() {
            const imageItems = document.querySelectorAll(".image-item");

            imageItems.forEach(imageItem => {
                const img = imageItem.querySelector("img");
                const videoNameSpan = imageItem.querySelector(".video-name");

                const imageName = img.getAttribute("alt");
                const videoName = imageName.split(" ")[1]; // Trích xuất tên video từ tên ảnh
                videoNameSpan.textContent = videoName;
            });
        });

        const outerColumnImages = document.querySelectorAll('.outer-column .image-item');

        outerColumnImages.forEach(item => {
            item.addEventListener('mouseenter', () => {
                item.style.transform = 'scale(3)'; // Điều chỉnh tỷ lệ phóng to theo nhu cầu
                item.style.zIndex = '1';
            });

            item.addEventListener('mouseleave', () => {
                item.style.transform = 'none';
                item.style.zIndex = 'auto';
            });
        });
        */

        // Lấy trường nhập liệu có ID là "textImage"
        var inputTextImage = document.getElementById("textImage");

        // Lấy trường nhập liệu có ID là "urlImage"
        var inputUrlImage = document.getElementById("urlImage");

        // Lấy trường có ID là "queryOutput"
        var queryOutput = document.getElementById("queryOutput");

        // Lưu giá trị của trường nhập liệu "textImage" vào localStorage
        inputTextImage.value = getSavedValue("textImage");

        // Lưu giá trị của trường nhập liệu "urlImage" vào localStorage
        inputUrlImage.value = getSavedValue("urlImage");

        // Hàm để lưu giá trị vào localStorage dựa trên ID của trường nhập liệu và giá trị nhập vào
        function saveValue(e){
            var id = e.id;
            var val = e.value;
            localStorage.setItem(id, val);
        }

        // Hàm để lấy giá trị đã lưu từ localStorage dựa trên khóa "v"
        function getSavedValue(v){
            if (!localStorage.getItem(v)) {
                return "";
            }
            return localStorage.getItem(v);
        }
        document.querySelector(".search-form").addEventListener("submit", function (event) {
            event.preventDefault();
            const query = document.getElementById("textImage").value ||
                        document.getElementById("urlImage").value ||
                        document.getElementById("speechToText").value ||
                        document.getElementById("objectCount").value;
            document.getElementById("queryOutput").textContent = "<strong>Query:</strong> " + query;
        });

        document.querySelector(".search-form").addEventListener("reset", function () {
        document.getElementById("queryOutput").textContent = "<strong>Query:</strong>";
        });