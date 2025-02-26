// Добавление комментариев
const commentForm = document.getElementById('comment-form');
const commentInput = document.getElementById('comment-input');
const commentsList = document.getElementById('comments-list');

commentForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const commentText = commentInput.value.trim();

    if (commentText) {
        const commentDiv = document.createElement('div');
        commentDiv.classList.add('comment');
        commentDiv.innerHTML = `<p>${commentText}</p>`;
        commentsList.appendChild(commentDiv);
        commentInput.value = '';
    }
});

// Оценка видео
const stars = document.querySelectorAll('.star');
const selectedRating = document.getElementById('selected-rating');

stars.forEach(star => {
    star.addEventListener('click', function () {
        const value = this.getAttribute('data-value');
        selectedRating.textContent = value;

        // Подсветка выбранных звезд
        stars.forEach((s, index) => {
            if (index < value) {
                s.classList.add('active');
            } else {
                s.classList.remove('active');
            }
        });
    });
});