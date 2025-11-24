"""
Sample Django views for testing.
"""

from django.views import View
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response


class ArticleListView(View):
    """List and create articles"""

    def get(self, request):
        """Get all articles"""
        articles = [
            {'id': 1, 'title': 'First Article'},
            {'id': 2, 'title': 'Second Article'},
        ]
        return JsonResponse({'articles': articles})

    def post(self, request):
        """Create new article"""
        title = request.POST.get('title')
        return JsonResponse({'id': 3, 'title': title})


@api_view(['GET', 'POST'])
def user_list(request):
    """List users or create new user"""
    if request.method == 'GET':
        return Response({'users': []})
    return Response({'created': True})
