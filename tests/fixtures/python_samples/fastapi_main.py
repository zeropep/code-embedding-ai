"""
Sample FastAPI application for testing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional


app = FastAPI(title="Sample API")


class Item(BaseModel):
    """Item schema"""
    id: int
    name: str
    description: Optional[str] = None
    price: float


class ItemCreate(BaseModel):
    """Schema for creating items"""
    name: str
    description: Optional[str] = None
    price: float


items_db: List[Item] = []


@app.get("/items/", response_model=List[Item])
async def get_items():
    """Get all items"""
    return items_db


@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get item by ID"""
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):
    """Create new item"""
    new_item = Item(
        id=len(items_db) + 1,
        name=item.name,
        description=item.description,
        price=item.price
    )
    items_db.append(new_item)
    return new_item
