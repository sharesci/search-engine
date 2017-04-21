import { Component, Input } from '@angular/core';

@Component({
    selector: 'ss-navbar',
    templateUrl: 'src/app/navbar/navbar.component.html'
})

export class NavbarComponent {
    @Input() ishome: string;
}