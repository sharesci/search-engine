import { Component } from '@angular/core';
import { SharedService } from '../../services/shared.service.js';

@Component({
    templateUrl: 'src/app/components/profile/profile.component.html',
    styleUrls: ['src/app/components/profile/profile.component.css']
})

export class ProfileComponent {
    constructor(private _sharedService: SharedService) { }

}