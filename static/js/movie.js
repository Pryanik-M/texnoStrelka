document.addEventListener('DOMContentLoaded', function() {
    const stars = document.querySelectorAll('.star');
    const selectedRating = document.getElementById('selected-rating');

    // Функция для подсветки звезд
    function highlightStars(rating) {
        stars.forEach((star, index) => {
            if (index < rating) {
                star.style.color = '#ffcc00'; // Желтый цвет для подсветки
            } else {
                star.style.color = '#ccc'; // Серый цвет по умолчанию
            }
        });
    }

    // Обработчик наведения на звезду
    stars.forEach(star => {
        star.addEventListener('mouseover', function() {
            const ratingValue = parseInt(this.getAttribute('data-value'), 10);
            highlightStars(ratingValue);
            selectedRating.textContent = ratingValue;
        });

        // Обработчик ухода курсора с области звезд
        star.addEventListener('mouseout', function() {
            const currentRating = parseInt(selectedRating.textContent, 10);
            highlightStars(currentRating);
        });

        // Обработчик нажатия на звезду
        star.addEventListener('click', function() {
            const ratingValue = parseInt(this.getAttribute('data-value'), 10);
            selectedRating.textContent = ratingValue;

            // Отправка данных на сервер
            fetch('/rate_movie', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    movie_id: 1, // Замените на реальный ID фильма
                    score: ratingValue
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Оценка сохранена!');
                } else {
                    alert('Ошибка при сохранении оценки.');
                }
            })
            .catch(error => {
                console.error('Ошибка:', error);
            });
        });
    });
});