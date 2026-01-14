from feast import Entity, ValueType

user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="User entity",
)
