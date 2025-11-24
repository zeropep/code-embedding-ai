"""
Sample Flask application for testing.
"""

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api/items', methods=['GET'])
def get_items():
    """Get all items"""
    items = [
        {'id': 1, 'name': 'Item 1'},
        {'id': 2, 'name': 'Item 2'},
    ]
    return jsonify(items)


@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    """Get single item by ID"""
    return jsonify({'id': item_id, 'name': f'Item {item_id}'})


@app.route('/api/items', methods=['POST'])
def create_item():
    """Create new item"""
    data = request.get_json()
    return jsonify({'id': 3, 'name': data.get('name')}), 201


if __name__ == '__main__':
    app.run(debug=True)
