from __future__ import annotations

from collections import Counter

from .schema import AllocationResult, OptimizationInput


def validate_input(data: OptimizationInput) -> list[str]:
    errors: list[str] = []
    if not data.passengers:
        errors.append("At least one passenger is required.")
    if data.bus.capacity <= 0:
        errors.append("Bus capacity must be positive.")
    if data.bus.price < 0:
        errors.append("Bus price cannot be negative.")
    if data.bus.recommended_minimum_passengers <= 0:
        errors.append("Recommended minimum passengers must be positive.")
    if data.bus.maximum_buses is not None and data.bus.maximum_buses <= 0:
        errors.append("Maximum bus count must be positive.")

    reservation_ids = [passenger.reservation_id for passenger in data.passengers]
    duplicates = sorted(
        reservation_id
        for reservation_id, count in Counter(reservation_ids).items()
        if count > 1
    )
    if duplicates:
        errors.append(f"Duplicate reservation IDs: {', '.join(duplicates)}")

    for passenger in data.passengers:
        if not passenger.reservation_id.strip():
            errors.append("Every passenger needs a reservation ID.")
        if not passenger.campus.strip():
            errors.append(f"{passenger.reservation_id}: campus is required.")
        if not passenger.team.strip():
            errors.append(f"{passenger.reservation_id}: team is required.")
        if not passenger.first_choice.strip() or not passenger.second_choice.strip():
            errors.append(
                f"{passenger.reservation_id}: first and second choices are required."
            )
    return errors


def validate_result(data: OptimizationInput, result: AllocationResult) -> list[str]:
    if result.status != "OPTIMAL":
        return ["Only OPTIMAL results can be validated for draft creation."]

    errors: list[str] = []
    passenger_by_id = {
        passenger.reservation_id: passenger for passenger in data.passengers
    }
    bus_by_id = {bus.bus_id: bus for bus in result.buses}

    if len(bus_by_id) != len(result.buses):
        errors.append("Duplicate bus IDs exist.")
    if result.total_buses != len(result.buses):
        errors.append("Reported total bus count does not match physical buses.")
    if (
        data.bus.maximum_buses is not None
        and result.total_buses > data.bus.maximum_buses
    ):
        errors.append("Reported total bus count exceeds the configured maximum.")
    if result.total_cost != result.total_buses * data.bus.price:
        errors.append("Reported total cost does not match bus count and price.")

    assignment_counts = Counter(
        assignment.reservation_id for assignment in result.assignments
    )
    expected_ids = set(passenger_by_id)
    assigned_ids = set(assignment_counts)
    if expected_ids != assigned_ids:
        errors.append("Assignments do not match the optimization passenger snapshot.")
    if any(count != 1 for count in assignment_counts.values()):
        errors.append("Every passenger must appear exactly once.")

    assignments_by_bus: dict[str, list[str]] = {}
    seats_by_bus: dict[str, list[int]] = {}
    second_choice_count = 0
    for assignment in result.assignments:
        passenger = passenger_by_id.get(assignment.reservation_id)
        bus = bus_by_id.get(assignment.bus_id)
        if passenger is None or bus is None:
            errors.append("Assignment references an unknown passenger or bus.")
            continue
        if assignment.destination != bus.destination:
            errors.append("Assignment destination does not match its bus destination.")
        if assignment.destination == passenger.first_choice:
            expected_rank = 1
        elif assignment.destination == passenger.second_choice:
            expected_rank = 2
            second_choice_count += 1
        else:
            expected_rank = 0
            errors.append(
                f"{assignment.reservation_id}: assignment is outside preferences."
            )
        if assignment.preference_rank != expected_rank:
            errors.append(
                f"{assignment.reservation_id}: reported preference rank is invalid."
            )
        assignments_by_bus.setdefault(assignment.bus_id, []).append(
            assignment.reservation_id
        )
        seats_by_bus.setdefault(assignment.bus_id, []).append(assignment.seat_number)

    if second_choice_count != result.second_choice_count:
        errors.append("Reported second-choice count is invalid.")

    for bus in result.buses:
        assigned = assignments_by_bus.get(bus.bus_id, [])
        if bus.capacity != data.bus.capacity:
            errors.append(f"{bus.bus_id}: bus capacity does not match the input.")
        if bus.price != data.bus.price:
            errors.append(f"{bus.bus_id}: bus price does not match the input.")
        if len(assigned) == 0:
            errors.append(f"{bus.bus_id}: empty buses are not allowed.")
        if len(assigned) > data.bus.capacity:
            errors.append(f"{bus.bus_id}: bus capacity exceeded.")
        if tuple(sorted(assigned)) != tuple(sorted(bus.passenger_ids)):
            errors.append(f"{bus.bus_id}: passenger list does not match assignments.")
        seats = sorted(seats_by_bus.get(bus.bus_id, []))
        if seats != list(range(1, len(assigned) + 1)):
            errors.append(f"{bus.bus_id}: seat numbers must be contiguous and unique.")

    return sorted(set(errors))
